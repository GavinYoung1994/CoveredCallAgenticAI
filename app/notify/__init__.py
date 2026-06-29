"""Outbound notifications — the Human-In-The-Loop (HITL) checkpoint.

Per design §3, autonomous live trading is forbidden: the Risk Manager must send
its top candidates to a human (via Discord here) for final execution. This
package builds and delivers that summary.
"""

from app.notify.discord_webhook import DiscordNotifier, format_recommendations

__all__ = ["DiscordNotifier", "format_recommendations"]
