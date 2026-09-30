"""LS-Face UI Design System and Components."""
from pathlib import Path
from kivy.lang import Builder
from . import tokens
from .behaviors import AppActionCard, ElevatedCardBehavior, HoverBehavior

UI_PATH = Path(__file__).resolve().parent
COMPONENTS_KV = UI_PATH / "components.kv"
FONTS_DIR = UI_PATH.parent / "assets" / "fonts"


_design_system_loaded = False


def load_design_system():
    """Load fonts and design system components into Kivy."""
    global _design_system_loaded
    if _design_system_loaded:
        return
    _design_system_loaded = True

    from kivy.core.text import LabelBase
    from kivy.factory import Factory

    # Register reusable behaviors & card components
    Factory.register("HoverBehavior", cls=HoverBehavior)
    Factory.register("ElevatedCardBehavior", cls=ElevatedCardBehavior)
    Factory.register("AppActionCard", cls=AppActionCard)

    # Register Montserrat font family
    if (FONTS_DIR / "Montserrat-Regular.ttf").exists():
        LabelBase.register(
            name="Montserrat",
            fn_regular=str(FONTS_DIR / "Montserrat-Regular.ttf"),
            fn_bold=str(FONTS_DIR / "Montserrat-Bold.ttf"),
        )
        # Register as fallback default so all Kivy standard widgets adopt it
        LabelBase.register(
            name="Roboto",
            fn_regular=str(FONTS_DIR / "Montserrat-Regular.ttf"),
            fn_bold=str(FONTS_DIR / "Montserrat-Bold.ttf"),
        )

    if (FONTS_DIR / "Montserrat-Black.ttf").exists():
        LabelBase.register(
            name="MontserratBlack",
            fn_regular=str(FONTS_DIR / "Montserrat-Black.ttf"),
        )

    if (FONTS_DIR / "Montserrat-Medium.ttf").exists():
        LabelBase.register(
            name="MontserratMedium",
            fn_regular=str(FONTS_DIR / "Montserrat-Medium.ttf"),
        )

    # Register CJK font support for Japanese and Korean
    cjk_candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialuni.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf"),
    ]
    cjk_font = next((str(p) for p in cjk_candidates if p.exists()), None)
    if cjk_font:
        LabelBase.register(name="AppCJK", fn_regular=cjk_font)
        tokens.FONT_CJK = "AppCJK"
    else:
        tokens.FONT_CJK = tokens.FONT_REGULAR

    if COMPONENTS_KV.exists():
        Builder.load_file(str(COMPONENTS_KV))


__all__ = [
    "tokens",
    "HoverBehavior",
    "ElevatedCardBehavior",
    "AppActionCard",
    "load_design_system",
    "UI_PATH",
    "COMPONENTS_KV",
    "FONTS_DIR",
]
