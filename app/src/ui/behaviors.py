"""LS-Face UI Reusable Behaviors and Interactive Components."""
from __future__ import annotations

from kivy.animation import Animation
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, StringProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout

from . import tokens


class HoverBehavior:
    """Reusable behavior for mouse hover detection and cursor styling in Kivy.

    Tracks Window mouse position safely across screen transitions and modal overlays.
    """

    hovered = BooleanProperty(False)
    hover_cursor = StringProperty("hand")
    hover_enabled = BooleanProperty(True)

    _active_hover_count = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self._on_mouse_motion)
        self.bind(parent=self._on_parent_change)

    def _on_parent_change(self, _widget, parent):
        if parent is None and self.hovered:
            self.hovered = False

    def _on_mouse_motion(self, _window, pos):
        if not self.hover_enabled or not self.get_root_window():
            if self.hovered:
                self.hovered = False
            return

        if self.opacity <= 0.0 or self.disabled:
            if self.hovered:
                self.hovered = False
            return

        # Check collision in widget coordinates
        try:
            inside = self.collide_point(*self.to_widget(*pos))
        except Exception:
            inside = False

        if inside != self.hovered:
            self.hovered = inside

    def on_hovered(self, _instance, value: bool):
        if value:
            HoverBehavior._active_hover_count += 1
            if self.hover_cursor:
                try:
                    Window.set_system_cursor(self.hover_cursor)
                except Exception:
                    pass
            self.on_hover_enter()
        else:
            HoverBehavior._active_hover_count = max(0, HoverBehavior._active_hover_count - 1)
            if HoverBehavior._active_hover_count == 0:
                try:
                    Window.set_system_cursor("arrow")
                except Exception:
                    pass
            self.on_hover_leave()

    def on_hover_enter(self):
        """Hook for hover enter event."""
        pass

    def on_hover_leave(self):
        """Hook for hover leave event."""
        pass


class ElevatedCardBehavior(HoverBehavior):
    """Reusable behavior providing soft drop shadow depth and hover micro-lift.

    Default rest state:
      - 0 1px 3px rgba(0, 0, 0, 0.06) soft shadow
      - 1px border, 16px radius

    Hover state:
      - translateY -2px micro-lift (lift_amount = dp(2))
      - deepened shadow: 0 3px 8px rgba(0, 0, 0, 0.12)
    """

    is_primary = BooleanProperty(False)
    badge_text = StringProperty("")
    lift_amount = NumericProperty(dp(tokens.HOVER_LIFT_Y))
    anim_duration = NumericProperty(0.16)

    hover_lift = NumericProperty(0.0)
    shadow_blur = NumericProperty(dp(tokens.SHADOW_BLUR_REST))
    shadow_offset_y = NumericProperty(dp(tokens.SHADOW_OFFSET_REST_Y))
    shadow_color = ListProperty(list(tokens.SHADOW_COLOR_SOFT))

    def on_hover_enter(self):
        if self.state == "down":
            return
        self._animate_elevation(
            lift=self.lift_amount,
            blur=dp(tokens.SHADOW_BLUR_HOVER),
            offset_y=dp(tokens.SHADOW_OFFSET_HOVER_Y),
            color=list(tokens.SHADOW_COLOR_HOVER),
        )

    def on_hover_leave(self):
        if self.state == "down":
            return
        self._animate_elevation(
            lift=0.0,
            blur=dp(tokens.SHADOW_BLUR_REST),
            offset_y=dp(tokens.SHADOW_OFFSET_REST_Y),
            color=list(tokens.SHADOW_COLOR_SOFT),
        )

    def on_state(self, _instance, value: str):
        """Support ButtonBehavior press / down feedback."""
        if value == "down":
            self._animate_elevation(
                lift=0.0,
                blur=dp(tokens.SHADOW_BLUR_REST * 0.75),
                offset_y=dp(tokens.SHADOW_OFFSET_REST_Y * 0.5),
                color=list(tokens.SHADOW_COLOR_PRESSED),
                duration=0.08,
            )
        else:
            if self.hovered:
                self.on_hover_enter()
            else:
                self.on_hover_leave()

    def _animate_elevation(self, lift: float, blur: float, offset_y: float, color: list, duration: float | None = None):
        d = duration if duration is not None else self.anim_duration
        Animation.stop_all(self, "hover_lift", "shadow_blur", "shadow_offset_y", "shadow_color")
        anim = Animation(
            hover_lift=lift,
            shadow_blur=blur,
            shadow_offset_y=offset_y,
            shadow_color=color,
            d=d,
            t="out_quad",
        )
        anim.start(self)


class AppActionCard(ElevatedCardBehavior, ButtonBehavior, BoxLayout):
    """Interactive navigation dashboard action card with depth and hover micro-lift."""

    card_title = StringProperty("")
    card_body = StringProperty("")
    icon_source = StringProperty("")
    inverted = BooleanProperty(False)

    def on_inverted(self, _instance, value: bool):
        # Backward compatibility for existing references to inverted
        if value:
            self.is_primary = True
