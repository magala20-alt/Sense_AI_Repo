# app_state.py — global state shared across all screens

from Backend.database.db_manager import DBManager


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
        self.backend = None
        self.conv_history = []

        # Single DB access point for desktop app flows
        self.db = DBManager()

    def set_authenticated_user(self, user: dict, token: str | None = None):
        if user and "full_name" not in user:
            # normalize UI key from DB 'username'
            user = {
                "id": user.get("id"),
                "full_name": user.get("username", ""),
                "email": user.get("email", ""),
            }
        self.current_user = user
        self.session_token = token

    def add_message(self, who: str, text: str, grammar_type: str = ""):
        self.conv_history.append({"who": who, "text": text, "grammar_type": grammar_type})

    def clear_history(self):
        self.conv_history = []

    def register_user(
        self,
        full_name,
        email,
        password,
        security_question=None,
        security_answer=None,
    ):
        user = self.db.create_user(
            username=(full_name or "").strip(),
            email=(email or "").strip(),
            password=password,
            security_question=security_question or "",
            security_answer=security_answer or "",
        )
        if not user:
            return False, "Account already exists."
        return True, "Account created successfully."

    def set_security_qa(self, email, question, answer):
        # Optional standalone update path
        q = (question or "").strip()
        a = (answer or "").strip()
        if not q or not a:
            return False, "Security question and answer are required."

        # Reuse DB helper behavior through direct SQL method absence:
        # create_user already stores QA; this method supports later edits.
        # If you want a dedicated DBManager method, add one and call it here.
        sec_q = self.db.get_security_question(email)
        if sec_q is None:
            return False, "Account not found."

        ok = self.db.verify_security_answer(email, a)  # quick check for same answer
        if ok and sec_q == q:
            return True, "Security question saved."

        # Minimal fallback: reset password API exists; no direct QA update method yet.
        # Recommend adding DBManager.update_security_qa(...) later.
        return False, "Add DBManager.update_security_qa(...) to support editing."

    def get_security_question(self, email):
        return self.db.get_security_question(email)

    def verify_security_answer(self, email, answer):
        return self.db.verify_security_answer(email, answer)

    def update_password(self, email, new_password):
        ok = self.db.update_password_by_email(email, new_password)
        if not ok:
            return False, "Could not update password."
        return True, "Password updated. Please sign in."

    def authenticate_user(self, email: str, password: str):
        user = self.db.authenticate_user_by_email(email, password)
        if not user:
            return False, "Invalid email or password."

        token = self.db.create_session_token(user.id)
        self.set_authenticated_user(
            {"id": user.id, "username": user.username, "email": user.email},
            token=token,
        )
        return True, "Login successful."

    def logout(self):
        if self.session_token:
            self.db.invalidate_session_token(self.session_token)
        self.current_user = None
        self.session_token = None
        self.user_role = ""
        self.session_id = ""
