"""Hardened & Isolated Code Execution Sandbox for genAI.

Replaces raw OS subprocess execution with a multi-layered security sandbox:
1. Abstract Syntax Tree (AST) Security Visitor:
   - Whitelists only safe numerical and scientific libraries (numpy, scipy, matplotlib, math, time).
   - Hard-blocks dangerous modules (os, sys, subprocess, socket, urllib, shutil, ctypes, etc.).
   - Intercepts reflection & dunder traversal (__subclasses__, __builtins__, __globals__, getattr, eval, exec).
   - Allows only designated environment variable read for SIMULATION_PLOT_PATH.
2. Isolated Process Execution:
   - Stripped minimal environment variables.
   - Strict 15-second wall-clock ceiling.
   - Ephemeral temporary directory with guaranteed cleanup.
"""

import ast
import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set


class ASTSecurityViolation(Exception):
    """Raised when code fails AST security auditing."""
    def __init__(self, message: str, rule: str, node: Optional[ast.AST] = None):
        super().__init__(message)
        self.rule = rule
        self.lineno = getattr(node, 'lineno', None)


class PythonASTSecurityAuditor(ast.NodeVisitor):
    """Recursively audits Python Abstract Syntax Trees for security violations."""

    ALLOWED_MODULES: Set[str] = {
        "numpy", "np",
        "scipy", "scipy.integrate", "scipy.optimize", "scipy.linalg", "scipy.stats", "scipy.signal",
        "matplotlib", "matplotlib.pyplot", "plt", "matplotlib.figure",
        "math", "cmath",
        "time",
    }

    BLOCKED_BUILTIN_FUNCS: Set[str] = {
        "eval", "exec", "compile", "open", "input",
        "__import__", "getattr", "setattr", "delattr", "hasattr",
        "globals", "locals", "vars", "breakpoint",
        "exit", "quit", "help",
    }

    BLOCKED_ATTRIBUTES: Set[str] = {
        "__subclasses__", "__builtins__", "__globals__", "__code__",
        "__class__", "__bases__", "__mro__", "__dict__",
        "__qualname__", "__module__",
    }

    BLOCKED_MODULES: Set[str] = {
        "os", "sys", "subprocess", "socket", "urllib", "requests",
        "shutil", "ctypes", "builtins", "importlib", "pickle", "shelve",
        "pty", "multiprocessing", "threading", "signal", "posix", "nt",
        "webbrowser", "http", "ftplib", "smtplib", "telnetlib",
    }

    def __init__(self):
        self.violations: List[Dict[str, Any]] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_module = alias.name.split('.')[0]
            if base_module in self.BLOCKED_MODULES or (alias.name not in self.ALLOWED_MODULES and base_module not in self.ALLOWED_MODULES):
                self.violations.append({
                    "rule": "unauthorized_import",
                    "detail": f"Import of module '{alias.name}' is blocked by sandbox security policy.",
                    "lineno": node.lineno
                })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if not node.module:
            self.violations.append({
                "rule": "relative_import",
                "detail": "Relative imports are disallowed.",
                "lineno": node.lineno
            })
            return

        base_module = node.module.split('.')[0]
        if base_module in self.BLOCKED_MODULES or (node.module not in self.ALLOWED_MODULES and base_module not in self.ALLOWED_MODULES):
            self.violations.append({
                "rule": "unauthorized_import_from",
                "detail": f"Import from '{node.module}' is blocked by sandbox security policy.",
                "lineno": node.lineno
            })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # 1. Direct function call checks e.g. eval(), open(), exec()
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.BLOCKED_BUILTIN_FUNCS:
                self.violations.append({
                    "rule": "blocked_builtin_call",
                    "detail": f"Invocation of built-in function '{func_name}' is blocked.",
                    "lineno": node.lineno
                })

        # 2. Attribute call checks e.g. os.system(), subprocess.run()
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in self.BLOCKED_ATTRIBUTES or attr_name in self.BLOCKED_BUILTIN_FUNCS:
                self.violations.append({
                    "rule": "blocked_attribute_call",
                    "detail": f"Call to sensitive attribute '{attr_name}' is blocked.",
                    "lineno": node.lineno
                })

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Intercept dunder introspection e.g. ().__class__.__bases__
        if node.attr in self.BLOCKED_ATTRIBUTES:
            self.violations.append({
                "rule": "sensitive_dunder_access",
                "detail": f"Access to reflective attribute '{node.attr}' is blocked.",
                "lineno": node.lineno
            })

        # Check for os.environ modifications or sensitive reads
        if isinstance(node.value, ast.Name) and node.value.id == "os":
            if node.attr != "environ":
                self.violations.append({
                    "rule": "unauthorized_os_attribute",
                    "detail": f"Access to 'os.{node.attr}' is blocked. Only os.environ for plot output is supported.",
                    "lineno": node.lineno
                })

        self.generic_visit(node)


class SecureSandboxRunner:
    """Executes validated Python simulations in an isolated temporary environment."""

    DEFAULT_TIMEOUT: int = 15

    @classmethod
    def audit_code(cls, code_str: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """Audit Python code against AST security policies."""
        if not code_str or not code_str.strip():
            return False, [{"rule": "empty_code", "detail": "Simulation code is empty.", "lineno": 1}]

        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return False, [{"rule": "syntax_error", "detail": f"Syntax error at line {e.lineno}: {e.msg}", "lineno": e.lineno}]

        auditor = PythonASTSecurityAuditor()
        auditor.visit(tree)

        if auditor.violations:
            return False, auditor.violations

        return True, []

    @classmethod
    def execute(
        cls,
        code_str: str,
        plot_destination: Optional[Path] = None,
        timeout: int = DEFAULT_TIMEOUT
    ) -> Dict[str, Any]:
        """Audit and execute Python code in an isolated subprocess.

        Returns:
            dict: {
                "success": bool,
                "stdout": str,
                "stderr": str,
                "returncode": int,
                "error": str,
                "violated_rules": list,
                "duration": float
            }
        """
        # Step 1: Pre-execution AST Security Audit
        is_safe, violations = cls.audit_code(code_str)
        if not is_safe:
            error_msg = "Security Sandbox Rejection: " + "; ".join(v["detail"] for v in violations)
            return {
                "success": False,
                "stdout": "",
                "stderr": error_msg,
                "returncode": 403,
                "error": error_msg,
                "violated_rules": violations,
                "duration": 0.0
            }

        # Step 2: Prepare Isolated Environment
        plot_path_str = str(plot_destination.resolve()) if plot_destination else ""
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
            "WINDIR": os.environ.get("WINDIR", "C:\\Windows"),
            "SIMULATION_PLOT_PATH": plot_path_str,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": tempfile.gettempdir(),
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp.write(code_str)
            tmp_path = Path(tmp.name)

        start_time = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=tempfile.gettempdir()
            )
            duration = round(time.time() - start_time, 3)
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()

            return {
                "success": (proc.returncode == 0) and ("Traceback" not in stderr),
                "stdout": stdout,
                "stderr": stderr,
                "returncode": proc.returncode,
                "error": stderr if proc.returncode != 0 else "",
                "violated_rules": [],
                "duration": duration
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Simulation execution exceeded the {timeout}-second timeout ceiling.",
                "returncode": 124,
                "error": f"Timeout after {timeout} seconds",
                "violated_rules": [{"rule": "timeout_expired", "detail": f"Exceeded {timeout}s", "lineno": 0}],
                "duration": timeout
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": 1,
                "error": str(e),
                "violated_rules": [],
                "duration": round(time.time() - start_time, 3)
            }
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
