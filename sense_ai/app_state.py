# app_state.py — global state shared across all screens

class AppState:
    """
    Global state shared across all screens.
    Pass this object to every screen constructor.
    """
    def __init__(self):
        self.user_role     = None          # "signer" or "speaker"
        self.session_id    = None          # conversation session ID string
        self.conv_history  = []            # list of {who, text, grammar_type}
        self.is_connected  = False         # WebSocket connection status
        self.current_translation = ""
        self.current_grammar     = ""
        self.tier_scores = {
            "physical": 0,
            "grammar":  0,
            "semantic": 0,
        }
    
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
