# UI/UX Design Brief & Design System
## genAI: Minimalist Cyber-Academic Design System

- **Document Version:** 2.4.0
- **Aesthetic Benchmark:** Linear + Perplexity Pro + Claude 3.5 Sonnet
- **Primary Design Mode:** Dark Mode Native (`#0c0e12`)
- **Status:** Approved / Active

---

## 1. Design Philosophy & Guiding Principles

1. **Zero-Clutter Information Density:**
   Academic and technical research requires high information density without visual chaos. Every border, badge, and interaction element must have a clear purpose.
2. **Content-First Canvas:**
   The research findings and formulas take visual primacy. UI controls, agent pods, and telemetry fade gracefully into secondary positions when reading.
3. **Dual-Theme Visual Identity:**
   - **Fast Chat:** Electric Cyan (`#22d3ee`) symbolizing speed, agility, and instant response.
   - **Deep Search:** Cosmic Violet & Royal Indigo (`#a855f7`, `#6366f1`) symbolizing deep exploration, multi-agent rigor, and cosmic research depth.
4. **Authentic Tactile Physics:**
   Buttons feature active click compressions (`active:scale-95`), floating search bars have backdrop blurs (`backdrop-blur-xl`), and tokens stream via a fluid 100fps typewriter pacing.

---

## 2. Color System & Design Tokens

```
┌────────────────────────────────────────────────────────────────────────┐
│                        COLOR PALETTE TOKENS                            │
└────────────────────────────────────────────────────────────────────────┘

  Background Surfaces:
  ├── Canvas Background:       #0c0e12  (Deep Obsidian Void)
  ├── Secondary Elevation:     #080a0f  (Dark Slate Basement)
  ├── Card Elevation:          #11141c  (Subtle Card Surface, border: #1d2230)
  └── Glass Search Float:      rgba(15, 18, 26, 0.85) (backdrop-blur-xl)

  Fast Chat Palette (Electric Cyan):
  ├── Primary Accent:          #22d3ee  (Tailwind cyan-400)
  ├── Hover State:             #67e8f9  (Tailwind cyan-300)
  ├── Ambient Background:      rgba(6, 182, 212, 0.12)
  └── Border Subtle:           rgba(6, 182, 212, 0.30)

  Deep Search Palette (Cosmic Violet & Indigo):
  ├── Gradient Start:          #7c3aed  (Tailwind violet-600)
  ├── Gradient Middle:         #a855f7  (Tailwind purple-500)
  ├── Gradient End:            #4f46e5  (Tailwind indigo-600)
  ├── Ambient Aura Glow:       rgba(168, 85, 247, 0.35) (shadow-[0_0_25px])
  └── Active Ring:             rgba(168, 85, 247, 0.50)

  Operational State Tokens:
  ├── Success / Verified:      #10b981  (Emerald green: Verified ✓, WAL Active)
  ├── Warning / Epistemic:     #f59e0b  (Amber: > ⚠️ Epistemic Note disclosures)
  ├── Stopped / Abort State:   #ef4444  (Crimson Red: ⏹ Stopped badge, Red Stop Block)
  └── Muted / Secondary Text:  #94a3b8  (Slate 400 for metadata & timestamps)
```

---

## 3. Typography Hierarchy

| Style Token | Font Family | Size | Weight | Line Height | Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Display Heading** | Inter / System | 32px (`text-3xl sm:text-4xl`) | Extra Bold (800) | Tight | Search Hero "Where research begins." |
| **Section Heading** | Inter / System | 20px (`text-xl`) | Bold (700) | Snug | Research section headings (`##`) |
| **Subsection Heading**| Inter / System | 16px (`text-base`) | Semi-Bold (600) | Normal | Research sub-headings (`###`) |
| **Body Prose** | Inter / System | 14px–15px (`text-sm sm:text-base`) | Regular (400) | 1.65 (Relaxed) | Main conversational answer markdown |
| **Code & Equations** | JetBrains Mono / monospace | 13px (`text-xs sm:text-sm`) | Medium (500) | Normal | LaTeX expressions, Python snippets |
| **Telemetry & Badges**| JetBrains Mono / monospace | 10px–11px (`text-[10px]`) | Bold (700) | Tight | Agent status pills, citations `[n]`, timers |

---

## 4. Core Component Specifications

### 4.1 Floating Bottom Search Bar (`#searchBarContainer`)
- **Structure:** Maximum width `max-w-4xl`, centered, floating above the bottom viewport with a gradient fade backdrop (`from-[#0c0e12] via-[#0c0e12]/95 to-transparent`).
- **Child Elements (Left to Right):**
  1. **Upload Paperclip Button (`#uploadPaperBtn`):**
     - Icon: Minimalist paperclip SVG.
     - Size: `36x36px` (`w-9 h-9 rounded-xl`).
     - States: Hover highlights border to cyan; during PDF ingestion, swaps icon to a spinning `✦` indicator.
  2. **Prompt Text Input (`#promptInput`):**
     - Borderless, transparent, auto-expanding text input with crisp slate placeholder text.
     - Adapts placeholder dynamically based on active mode:
       - *Fast Chat:* `"Ask anything... (Fast Chat like ChatGPT / DeepSeek)"`
       - *Deep Search:* `"Deep Search active: Enter any research inquiry, complex hypothesis, or technical question..."`
  3. **Cosmic Deep Search Pill (`#deepResearchToggleBtn`):**
     - Dual-state toggle button.
     - *OFF State:* Subtle dark glass pill with purple border and 4-point lens star.
     - *ON State:* Transforms into a radiant glowing violet/indigo gradient button with an active pinging live indicator dot (`scale-[1.02] shadow-[0_0_20px_rgba(168,85,247,0.5)]`).
  4. **Icon-Only Action Button (`#searchBtn`):**
     - Size: `36x36px` (`w-9 h-9 rounded-xl`).
     - Strictly icon-only (zero text clutter).
     - *Idle State:* High-contrast black up-arrow inside cyan (or purple) background.
     - *Generating State:* Transforms into the **Antigravity Red Stop Block**: Dark crimson button with an animated pinging red square indicator (`⏹`), allowing one-click cancellation.

### 4.2 Attached Document Chips (`#uploadedPapersList`)
- Located directly above the floating search bar.
- Stays `hidden` when empty.
- When PDFs are attached, renders dismissible pills:
  `[ 📑 paper_title.pdf (14p) ✕ ]` with backdrop blur, preventing screen clutter.

### 4.3 Multi-Agent Research Bar & Collapsible Drawer
- **Header Status Bar:** Compact `42px` bar showing agent handoffs, indexed source counts (`8 Sources`), and peer-review certification (`Verified ✓`).
- **Expandable Drawer:**
  - 5-column grid displaying the active specialist agent pods:
    - 🔬 Dr. Aris Thorne (Research Scout)
    - ⚖️ Prof. Elena Rostova (Adversarial Critic)
    - ✍️ Dr. Marcus Vance (Lead Synthesizer)
    - ✉️ Sophia Chen (Executive Comms)
    - 📅 David Sterling (Logistics)
  - Visual Stepper: 5 certified stages with green checkmarks upon verification completion.

### 4.4 Interactive Citation Badges & Popovers
- Rendered inline as high-contrast pill badges: `[1]`, `[2]`.
- **Hover Micro-Card:** Hovering over a citation pill opens a floating preview popover showing:
  - Source Title & Year.
  - Domain Badge (e.g., `arxiv.org`, `nature.com`).
  - Extracted Evidence Snippet.
- **Click Action:** Clicking opens the original publication or DOI in a guaranteed new tab (`target="_blank" rel="noopener noreferrer"`).

---

## 5. Responsive Breakpoint Matrix

| Viewport | Layout Adjustments |
| :--- | :--- |
| **Desktop (> 1024px)** | Full navigation sidebar visible, 5-column agent pods grid, centered 896px reading container (`max-w-4xl`). |
| **Tablet (768px - 1023px)** | Sidebar collapsible into hamburger overlay, agent pods render in a 3-column / 2-column wrapping layout. |
| **Mobile (< 768px)** | Hamburger icon toggles slide-out drawer, bottom search bar stretches edge-to-edge with 16px margins, Deep Search pill switches to icon-only mode to preserve input width. |
