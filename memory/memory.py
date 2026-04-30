from langchain_core.chat_history import InMemoryChatMessageHistory

class ChatMemory:
    def __init__(self, k=5):
        self.k = k
        self.store = InMemoryChatMessageHistory()

    def add(self, user, assistant):
        self.store.add_user_message(user)
        self.store.add_ai_message(assistant)

    def get_messages(self):
        return self.store.messages[-self.k*2:]

    def format(self):
        msgs = self.get_messages()
        return "\n".join([f"{m.type}: {m.content}" for m in msgs])