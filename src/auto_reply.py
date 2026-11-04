"""
auto_reply.py  (Bonus - Auto Reply Suggestion)
----------------------------------------------
A simple rule-based reply: each predicted category maps to a ready-made reply.
No API and no extra AI model is needed, so nothing can break and no keys are required.
"""

REPLIES = {
    "Login Issue": (
        "Hello, sorry you are having trouble signing in. Please try the 'Forgot Password' "
        "option on the login page to reset your password. If your account is locked, "
        "wait 15 minutes and try again, or reply to this message and we will unlock it for you."
    ),
    "Application Error": (
        "Hello, sorry about the error. Please refresh the page and try again. If the problem "
        "continues, reply with a screenshot of the error message and the steps that led to it, "
        "and our technical team will investigate."
    ),
    "Report": (
        "Hello, thanks for your request. Reports can be generated from the Reports section of "
        "the dashboard. If you need a custom report or a different date range, please share "
        "the details and we will prepare it for you."
    ),
    "Account Update": (
        "Hello, we can help with that. You can update your details under Profile > Settings. "
        "If you cannot find the option, reply with the change you need and we will update "
        "your account after verifying your identity."
    ),
    "Performance": (
        "Hello, sorry the system is running slowly. Please clear your browser cache and try "
        "again. If it is still slow, reply with the time it started and the page you were "
        "using, and our team will look into it."
    ),
}

DEFAULT_REPLY = (
    "Hello, thank you for contacting support. Your ticket has been received and a support "
    "agent will review it shortly."
)


def get_reply(category: str) -> str:
    """Return the suggested reply for a category (or a safe default)."""
    return REPLIES.get(category, DEFAULT_REPLY)