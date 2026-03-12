# app_state.py — global state shared across all screens

class AppState:
    """
    Global state shared across all screens.
    Pass this object to every screen constructor.
    """
    def __init__(self):
        self.current_user = None
        self.session_token = None
        self.user_role = ""
        self.session_id = ""

        # backend instance is attached in main.py:
        # self.state.backend = create_app()
        self.backend = None

        # conversation history used by signer/speaker screens
        self.conv_history = []

    def set_authenticated_user(self, user: dict, token: str | None = None):
        # backend returns username/email; keep UI compatibility
        if user and "full_name" not in user:
            user["full_name"] = user.get("username", "User")
        self.current_user = user
        self.session_token = token

    def add_message(self, who: str, text: str, grammar_type: str = ""):
        """Add a message to conversation history."""
        self.conv_history.append({
            "who": who,
            "text": text,
            "grammar_type": grammar_type
        })

    def clear_history(self):
        """Clear conversation history."""
        self.conv_history = []

    # Backward-compatible wrappers for screens still calling old methods
    def register_user(self, full_name: str, email: str, password: str):
        """
        Register using backend. Returns (success, message).
        full_name is mapped to username for current backend schema.
        """
        if not self.backend:
            return False, "Backend not initialized."

        username = (full_name or "").strip() or email.split("@")[0]
        result = self.backend.register(username=username, email=email, password=password)
        if not result.get("ok"):
            return False, result.get("error", "Registration failed.")

        # Auto-login after register
        login = self.backend.login(username=username, password=password)
        if login.get("ok"):
            self.set_authenticated_user(login["user"], login.get("token"))
            return True, "Account created successfully."

        return True, "Account created. Please log in."

    def authenticate_user(self, email: str, password: str):
        """
        Authenticate using backend. Returns (success, message).
        Existing UI passes email; backend login expects username.
        We try username = email first, then email-prefix fallback.
        """
        if not self.backend:
            return False, "Backend not initialized."

        candidates = [email.strip(), email.split("@")[0].strip()]
        for username in candidates:
            result = self.backend.login(username=username, password=password)
            if result.get("ok"):
                self.set_authenticated_user(result["user"], result.get("token"))
                name = self.current_user.get("full_name", "User").split()[0]
                return True, f"Welcome back, {name}."

        return False, "Invalid username/email or password."

    def logout(self):
        """Clear active user/session state."""
        if self.backend and self.session_token:
            try:
                self.backend.logout(self.session_token)
            except Exception:
                pass

        self.current_user = None
        self.session_token = None
        self.user_role = ""
        self.session_id = ""
