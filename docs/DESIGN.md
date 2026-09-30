# LS-Face UI Design System

**Status:** Active  
**Theme:** Primarily White (#FFFFFF / #F8FAFC), Accent La Salle Green (#006633), Black text (#0F151B) when necessary  
**Target:** 1280×720 Edge Kiosk & Desktop (Raspberry Pi Touchscreen compatible)  
**Implementation:** [tokens.py](file:///C:/Users/acer/Documents/USLS%204th%20Year/Computer%20Vision/porting-sets/lsface-raspberry-pi/app/src/ui/tokens.py) · [components.kv](file:///C:/Users/acer/Documents/USLS%204th%20Year/Computer%20Vision/porting-sets/lsface-raspberry-pi/app/src/ui/components.kv) · [__init__.py](file:///C:/Users/acer/Documents/USLS%204th%20Year/Computer%20Vision/porting-sets/lsface-raspberry-pi/app/src/ui/__init__.py)

---

## 1. Design Tokens

All tokens are defined in [src/ui/tokens.py](file:///C:/Users/acer/Documents/USLS%204th%20Year/Computer%20Vision/porting-sets/lsface-raspberry-pi/app/src/ui/tokens.py) as RGBA tuples normalized to `0.0 – 1.0` for Kivy graphics.

### 1.1 Color Tokens

#### Primary Brand Accent (La Salle Forest Green)
| Token | Hex | RGBA Value | Usage |
|---|---|---|---|
| `COLOR_ACCENT` | `#006633` | `(0.0, 0.40, 0.20, 1.0)` | Primary CTA buttons, active indicators, focus rings |
| `COLOR_ACCENT_HOVER` | `#007A3D` | `(0.0, 0.48, 0.24, 1.0)` | Button hover highlight, selected card border |
| `COLOR_ACCENT_PRESSED`| `#005229` | `(0.0, 0.32, 0.16, 1.0)` | Active pressed state |
| `COLOR_ACCENT_GLOW` | `#006633` | `(0.0, 0.40, 0.20, 0.18)`| Focus halo, selection background |
| `COLOR_ACCENT_TINT` | `#006633` | `(0.0, 0.40, 0.20, 0.08)`| Subtle pill/card wash & icon background |

#### Primarily White Base
| Token | Hex | RGBA Value | Usage |
|---|---|---|---|
| `COLOR_BG` | `#F8FAFC` | `(0.973, 0.980, 0.988, 1.0)` | Root screen canvas background |
| `COLOR_SURFACE_1` | `#FFFFFF` | `(1.0, 1.0, 1.0, 1.0)` | Pure white cards, panels, containers |
| `COLOR_SURFACE_2` | `#F1F4F7` | `(0.945, 0.957, 0.969, 1.0)` | Secondary controls, input backgrounds |
| `COLOR_SURFACE_3` | `#E5EAEF` | `(0.898, 0.918, 0.941, 1.0)` | Hover surfaces, pressed controls |
| `COLOR_VIEWPORT_BG` | `#0D1116`| `(0.050, 0.065, 0.085, 1.0)`| Frame surround for camera feed |

#### Borders & Dividers
| Token | Hex | RGBA Value | Usage |
|---|---|---|---|
| `COLOR_BORDER_MUTED` | `#EAEEF2` | `(0.918, 0.933, 0.949, 1.0)` | Table row dividers, hairline separators |
| `COLOR_BORDER` | `#D9DFE6` | `(0.851, 0.875, 0.902, 1.0)` | Card outline, container border |
| `COLOR_BORDER_LIGHT` | `#C6CFD9` | `(0.776, 0.812, 0.851, 1.0)` | Input border, interactive element borders |

#### High-Contrast Typography (Black when necessary)
| Token | Hex | RGBA Value | Usage |
|---|---|---|---|
| `COLOR_TEXT_PRIMARY` | `#0F151B` | `(0.060, 0.082, 0.106, 1.0)` | Headings, card titles, black text |
| `COLOR_TEXT_SECONDARY` | `#556271`| `(0.333, 0.384, 0.443, 1.0)` | Subtitles, body labels, secondary descriptions |
| `COLOR_TEXT_MUTED` | `#8C99A7` | `(0.550, 0.600, 0.655, 1.0)` | Captions, hints, footnotes |
| `COLOR_TEXT_ON_ACCENT`| `#FFFFFF`| `(1.0, 1.0, 1.0, 1.0)` | High-contrast white on green buttons |
| `COLOR_TEXT_DISABLED` | `#A0A8B4` | `(0.680, 0.720, 0.760, 1.0)` | Inactive actions |

#### Status & Diagnostics (Calibrated for white)
| Token | Hex | RGBA Value | Usage |
|---|---|---|---|
| `COLOR_SUCCESS` | `#008C45` | `(0.0, 0.550, 0.270, 1.0)` | LIVE badge, verified face, step completed |
| `COLOR_WARNING` | `#D97A0D` | `(0.850, 0.480, 0.050, 1.0)` | SETUP REQUIRED badge, calibration caution |
| `COLOR_ERROR` | `#D92E2E` | `(0.850, 0.180, 0.180, 1.0)` | Delete action, scan failure, bad frame |
| `COLOR_INFO` | `#1A73D9` | `(0.100, 0.450, 0.850, 1.0)` | Informational hints |

---

## 2. Metrics & Typography Scale

### 2.1 Spacing & Radius
- `RADIUS_SM`: `8dp` (Chips, small badges, table headers)
- `RADIUS_MD`: `12dp` (Buttons, inputs, dialog containers)
- `RADIUS_LG`: `16dp` (Cards, camera video viewports)
- `RADIUS_XL`: `20dp` (Home navigation action cards)
- `RADIUS_PILL`: `999dp` (Status badges)

### 2.2 Heights
- Header bar: `48dp`
- Standard action button: `40dp`
- Large primary CTA: `48dp`
- Input field / Spinner: `42dp`
- Table row: `48dp`

### 2.3 Typography Font Families & Scale
- **`tokens.FONT_BLACK` (`MontserratBlack`)**: `Montserrat-Black.ttf` — Used for high-impact hero titles (`Smart Gate`) and action card titles.
- **`tokens.FONT_BOLD` (`Montserrat`)**: `Montserrat-Bold.ttf` — Used for button labels, category tags, and active status badges.
- **`tokens.FONT_MEDIUM` (`MontserratMedium`)**: `Montserrat-Medium.ttf` — Used for subtitles and secondary headings.
- **`tokens.FONT_REGULAR` (`Montserrat`)**: `Montserrat-Regular.ttf` — Default text font (registered as standard Kivy `Roboto` fallback).

#### Scale
- **Hero Title**: `32sp`, Montserrat Black — Welcome title, screen hero
- **Card Title**: `22sp`, Montserrat Black — Action card navigation titles
- **Section Heading**: `17sp`, Montserrat Bold — Screen headers, modal headers
- **Body / Button**: `14sp`, Montserrat Medium / Regular — Buttons, descriptions
- **Caption / Badge**: `12sp`, Montserrat Bold / Regular — Pills, status notes, table cells
- **Micro**: `10sp`, Montserrat Bold Uppercase — Category tags, hardware telemetry

---

## 3. Component Catalog

All components are declared in [src/ui/components.kv](file:///C:/Users/acer/Documents/USLS%204th%20Year/Computer%20Vision/porting-sets/lsface-raspberry-pi/app/src/ui/components.kv) and loaded at application launch via `src.ui.load_design_system()`.

### 3.1 Buttons
| Component | Class Type | Description & Behavior |
|---|---|---|
| `PrimaryButton` | `ButtonBehavior+BoxLayout` | High-visibility CTA with `#006633` green fill, hover brightener, white text. Supports `button_text` and optional `icon_source`. |
| `SecondaryButton` | `ButtonBehavior+BoxLayout` | Dark surface button with `COLOR_BORDER_LIGHT` outline. Used for headers, auxiliary controls, and filters. |
| `DangerButton` | `ButtonBehavior+BoxLayout` | Dark crimson border and fill with light red text. Used for destructive actions (e.g. Delete Identity). |
| `BackButton` | `ButtonBehavior+BoxLayout` | Fixed `100dp × 40dp` back navigation button with `‹ Back` glyph and smooth press feedback. |
| `GhostButton` | `ButtonBehavior+BoxLayout` | Transparent background with hover highlight for non-critical cancel actions. |

**Example KV Usage:**
```kivy
PrimaryButton:
    button_text: 'Start Scan'
    icon_source: f"{ICONS_PATH}/scan.png"
    on_release: root.start_scan()

SecondaryButton:
    button_text: 'Calibrate Camera'
    on_release: root.calibrate()

DangerButton:
    button_text: 'Delete'
    on_release: root.confirm_delete()

BackButton:
    on_release: root.go_back()
```

### 3.2 Cards & Containers
| Component | Class Type | Description & Behavior |
|---|---|---|
| `AppCard` | `BoxLayout` | Surface container (`#FFFFFF`) with 16px radius (`RADIUS_LG`), 1px border (`COLOR_BORDER`), and soft depth shadow (`0 1px 3px rgba(0,0,0,0.06)`). |
| `AppActionCard` | `AppActionCard` (`ElevatedCardBehavior`, `ButtonBehavior`, `BoxLayout`) | Large dashboard navigation card with equalized visual weight, centered icon, bold title, description, and primary action indicators (`is_primary: True` or `badge_text`). Features interactive depth: rests with `0 1px 3px rgba(0,0,0,0.06)` shadow and lifts slightly on hover (`translateY -2px`, shadow deepens to `0 3px 8px rgba(0,0,0,0.12)`). |
| `VideoViewport` | `BoxLayout` | Camera container providing smooth rounded corners and dark border around the OpenCV stream. |

**Example KV Usage:**
```kivy
AppActionCard:
    is_primary: True
    badge_text: 'PRIMARY'
    card_title: 'Register'
    card_body: 'Register your face and try the system.'
    icon_source: f"{ICONS_PATH}/user-plus.png"
    on_release: root.open_add_identity()

AppActionCard:
    card_title: 'Live Recognition'
    card_body: 'Run real-time edge biometric verification.'
    icon_source: f"{ICONS_PATH}/scan-text.png"
    on_release: root.open_recognition()

VideoViewport:
    Image:
        id: camera_feed
        allow_stretch: True
        keep_ratio: True
```

### 3.3 Inputs & Selectors
| Component | Class Type | Description & Behavior |
|---|---|---|
| `AppTextInput` | `TextInput` | High-contrast dark input with `#006633` cursor and border glow on focus. |
| `AppSpinner` | `Spinner` | Dark dropdown with rounded corners and subtle border outline. |

**Example KV Usage:**
```kivy
AppTextInput:
    hint_text: 'Enter identity name'
    on_text: root.name = self.text

AppSpinner:
    text: 'Default PC Camera'
    values: CAMERA_NAMES
    on_text: root.on_camera_changed(self.text)
```

### 3.4 Status & Feedback
| Component | Class Type | Description & Behavior |
|---|---|---|
| `StatusBadge` | `BoxLayout` | Pill badge with colored dot. Accepts `badge_text` and `badge_status` (`'ready'`, `'live'`, `'warning'`, `'error'`, `'muted'`). |
| `StepIndicatorItem` | `BoxLayout` | Multi-step progress indicator (e.g. for FRONT / LEFT / RIGHT / UP / DOWN). States: `'waiting'`, `'active'`, `'done'`, `'error'`. |
| `AppProgressBar` | `ProgressBar` | Clean progress bar with `#006633` green fill and dark track. |

**Example KV Usage:**
```kivy
StatusBadge:
    badge_text: 'LIVE'
    badge_status: 'live'

StepIndicatorItem:
    label_text: 'FRONT'
    step_state: 'done'

AppProgressBar:
    max: 1
    value: root.progress
```

### 3.5 Data Tables
| Component | Class Type | Description & Behavior |
|---|---|---|
| `DataTableHeader` | `BoxLayout` | Elevated table header with bottom border stroke. |
| `DataTableRow` | `BoxLayout` | Table row container with subtle row separator line. |

---

## 4. How to Use in the App

### 4.1 In Python View Controllers
```python
from src.ui import tokens

# Example: setting dynamic label color
my_label.color = tokens.COLOR_SUCCESS
```

### 4.2 In KV Files
Components in `components.kv` are registered in Kivy's global factory via `load_design_system()` in `app/main.py`. Any KV file can directly instantiate:
```kivy
#:kivy 2.3.1
#:import tokens src.ui.tokens

<MyScreenView>:
    canvas.before:
        Color:
            rgba: tokens.COLOR_BG
        Rectangle:
            pos: self.pos
            size: self.size

    BoxLayout:
        orientation: 'vertical'
        padding: '24dp'
        spacing: '16dp'

        BackButton:
            on_release: root.go_back()

        AppCard:
            Label:
                text: 'Welcome to LS-Face'
                color: tokens.COLOR_TEXT_PRIMARY

        PrimaryButton:
            button_text: 'Confirm'
            on_release: root.confirm()
```

---

## 5. Screen Migration Map

The following table maps existing raw widgets to the new component system:

| Screen File | Legacy Raw Widget | New Design Component |
|---|---|---|
| `home.kv` | Raw `ActionCard` with inline colors | `AppActionCard` |
| `home.kv` | `HeaderButton`, `HeaderSpinner` | `SecondaryButton`, `AppSpinner` |
| `recognition.kv` | Inline `Button` '‹ Back' | `BackButton` |
| `recognition.kv` | Raw `Image` box | `VideoViewport` |
| `recognition.kv` | Hardcoded 'LIVE' label | `StatusBadge` (`badge_status: 'live'`) |
| `identities.kv` | Raw white canvas `(0.97, 0.97, 0.97, 1)` | `COLOR_BG`, `DataTableHeader`, `DataTableRow` |
| `identities.kv` | Raw default Kivy `Button` | `DangerButton` for Delete, `BackButton` for return |
| `pose.kv` | Inline `PoseStep` | `StepIndicatorItem` |
| `pose.kv` | Raw `ProgressBar`, inline `Button` | `AppProgressBar`, `PrimaryButton`, `SecondaryButton` |
| `voice_recognition.kv` | Inline `TextInput` & `ProgressBar` | `AppTextInput`, `AppProgressBar`, `PrimaryButton` |