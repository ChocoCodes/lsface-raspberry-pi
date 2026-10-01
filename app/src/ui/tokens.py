"""LS-Face Design System Tokens.

Theme:
- Primary: Pure White / Off-White (#FFFFFF / #F8FAFC)
- Accent: La Salle Forest Green (#006633)
- High Contrast Black (#0F151B) when necessary for text, icons, and sharp borders
"""
from __future__ import annotations

# ==============================================================================
# 1. COLOR TOKENS (RGBA tuples in 0.0 - 1.0 range for Kivy)
# ==============================================================================

# Accent - La Salle Green (#006633)
COLOR_ACCENT = (0.0, 0.40, 0.20, 1.0)           # #006633 primary brand green
COLOR_ACCENT_HOVER = (0.0, 0.48, 0.24, 1.0)     # #007A3D hover state
COLOR_ACCENT_PRESSED = (0.0, 0.32, 0.16, 1.0)   # #005229 active/press state
COLOR_ACCENT_GLOW = (0.0, 0.40, 0.20, 0.18)     # Focus halo / selection ring
COLOR_ACCENT_TINT = (0.0, 0.40, 0.20, 0.08)     # 8% subtle green wash for cards/pills
COLOR_ACCENT_BORDER = (0.0, 0.40, 0.20, 0.35)   # Border highlight

# Primarily White Base
COLOR_BG = (0.973, 0.980, 0.988, 1.0)           # #F8FAFC Screen canvas background
COLOR_SURFACE_1 = (1.0, 1.0, 1.0, 1.0)          # #FFFFFF Pure white cards & panels
COLOR_SURFACE_2 = (0.945, 0.957, 0.969, 1.0)    # #F1F4F7 Secondary controls & input fill
COLOR_SURFACE_3 = (0.898, 0.918, 0.941, 1.0)    # #E5EAEF Hover surface / active button
COLOR_SURFACE_HOVER = (0.930, 0.945, 0.960, 1.0)# #EDF1F5 Elevated card hover

# Borders & Dividers
COLOR_BORDER_MUTED = (0.918, 0.933, 0.949, 1.0) # #EAEEF2 Hairline divider
COLOR_BORDER = (0.851, 0.875, 0.902, 1.0)       # #D9DFE6 Default container border
COLOR_BORDER_LIGHT = (0.776, 0.812, 0.851, 1.0) # #C6CFD9 Interactive element border
COLOR_BORDER_DARK = (0.150, 0.180, 0.220, 1.0)  # #262E38 Black/charcoal accent border

# High Contrast Typography (Black when necessary)
COLOR_TEXT_PRIMARY = (0.060, 0.082, 0.106, 1.0) # #0F151B Near-pitch black
COLOR_TEXT_SECONDARY = (0.333, 0.384, 0.443, 1.0)# #556271 Dark slate body
COLOR_TEXT_MUTED = (0.550, 0.600, 0.655, 1.0)   # #8C99A7 Medium gray caption
COLOR_TEXT_ON_ACCENT = (1.0, 1.0, 1.0, 1.0)      # High-contrast white on green
COLOR_TEXT_DISABLED = (0.680, 0.720, 0.760, 1.0) # Inactive text

# Video Viewport Frame (Kept dark inside for letterbox / OpenCV contrast)
COLOR_VIEWPORT_BG = (0.050, 0.065, 0.085, 1.0)  # #0D1116 Dark frame surround

# Semantic Feedback (Calibrated for white backgrounds)
COLOR_SUCCESS = (0.0, 0.550, 0.270, 1.0)        # #008C45 Deep green (Live / Done)
COLOR_SUCCESS_TINT = (0.0, 0.550, 0.270, 0.10)  # 10% tint
COLOR_WARNING = (0.850, 0.480, 0.050, 1.0)      # #D97A0D Deep amber (Setup / Caution)
COLOR_WARNING_TINT = (0.850, 0.480, 0.050, 0.10) # 10% tint
COLOR_ERROR = (0.850, 0.180, 0.180, 1.0)        # #D92E2E Deep crimson (Failed / Delete)
COLOR_ERROR_TINT = (0.850, 0.180, 0.180, 0.10)  # 10% tint
COLOR_INFO = (0.100, 0.450, 0.850, 1.0)         # #1A73D9 Blue

COLOR_TRANSPARENT = (0.0, 0.0, 0.0, 0.0)

# ==============================================================================
# 2. METRIC TOKENS (Radii, Spacing, Heights)
# ==============================================================================
RADIUS_NONE = 0
RADIUS_XS = 4
RADIUS_SM = 8
RADIUS_MD = 12
RADIUS_LG = 16
RADIUS_XL = 20
RADIUS_PILL = 999

BORDER_WIDTH_THIN = 1.0
BORDER_WIDTH_FOCUS = 1.8
BORDER_WIDTH_THICK = 2.0

HEIGHT_HEADER = 48
HEIGHT_BUTTON_LG = 48
HEIGHT_BUTTON_MD = 40
HEIGHT_BUTTON_SM = 32
HEIGHT_INPUT = 42

FONT_SIZE_HERO = 32
FONT_SIZE_TITLE = 22
FONT_SIZE_SUBTITLE = 17
FONT_SIZE_BODY = 14
FONT_SIZE_CAPTION = 12
FONT_SIZE_MICRO = 10

# ==============================================================================
# 3. TYPOGRAPHY TOKENS (Montserrat Font Family)
# ==============================================================================
FONT_BLACK = "MontserratBlack"
FONT_TITLE = "MontserratBlack"
FONT_BOLD = "Montserrat"
FONT_MEDIUM = "MontserratMedium"
FONT_REGULAR = "Montserrat"
FONT_CJK = "AppCJK"

# ==============================================================================
# 4. SHADOW & ELEVATION TOKENS
# ==============================================================================
SHADOW_COLOR_SOFT = (0.0, 0.0, 0.0, 0.06)        # rgba(0, 0, 0, 0.06) rest shadow
SHADOW_COLOR_HOVER = (0.0, 0.0, 0.0, 0.12)       # rgba(0, 0, 0, 0.12) deepened hover shadow
SHADOW_COLOR_PRESSED = (0.0, 0.0, 0.0, 0.04)     # rgba(0, 0, 0, 0.04) pressed shadow

SHADOW_BLUR_REST = 3.0                           # 3px blur radius
SHADOW_BLUR_HOVER = 8.0                          # 8px blur radius
SHADOW_OFFSET_REST_Y = -1.0                      # -1px (downward)
SHADOW_OFFSET_HOVER_Y = -3.0                     # -3px (downward when lifted)
HOVER_LIFT_Y = 2.0                               # 2px translateY lift

