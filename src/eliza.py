"""Area B: ELIZA-style rule-based replies for greetings and small talk.

Like Weizenbaum's ELIZA (1966), each rule is a regular expression with reply
templates. A captured fragment is "reflected" (my -> your, I -> you) and
inserted into the reply. Small talk is answered here and never sent to a team.

reply(text)           -> a reply if a rule matches, else None
reply_for_intent(i)   -> a reply when a model predicts a small_talk intent
"""
import random
import re

REFLECTIONS = {"i": "you", "me": "you", "my": "your", "am": "are", "i'm": "you are",
               "mine": "yours", "myself": "yourself", "you": "I", "your": "my"}

RULES = [
    (r"^(hi|hello|hey|good (morning|afternoon|evening)|greetings)\b",
     ["Hello! I am the NovaAssist support bot. How can I help you today?",
      "Hi there! Tell me what you need help with."]),
    (r"\b(thanks|thank you|thx|cheers)\b",
     ["You're welcome! Anything else I can help with?"]),
    (r"^(bye|goodbye|see you|good night)\b",
     ["Goodbye! Have a great day."]),
    (r"\bhow are you\b",
     ["I'm doing well, thank you! What can I do for you?"]),
    (r"\bare you (a )?(bot|robot|human|real)\b",
     ["I am an automated assistant. If you need a person, just say 'human agent'."]),
    (r"\b(what is|what's) your name\b",
     ["I'm the NovaAssist support bot."]),
    (r"\btell me a joke\b",
     ["Why did the credit card break up with the wallet? It needed more space."]),
    (r"^i feel (.+)",
     ["Why do you feel {0}?", "How long have you felt {0}?"]),
    (r"^i am (sad|happy|bored|tired|angry|upset|frustrated)\b(.*)",
     ["I'm sorry to hear you are {0}. Is there something I can help you with?"]),
]
COMPILED = [(re.compile(p, re.I), r) for p, r in RULES]

INTENT_REPLIES = {
    "greeting": "Hello! How can I help you today?",
    "goodbye": "Goodbye! Have a great day.",
    "thank_you": "You're welcome!",
    "tell_joke": "Why did the credit card break up with the wallet? It needed more space.",
    "are_you_a_bot": "I am an automated assistant.",
    "what_is_your_name": "I'm the NovaAssist support bot.",
    "how_old_are_you": "I was launched this year, so I'm quite young!",
    "where_are_you_from": "I live in the NovaAssist cloud.",
    "who_made_you": "I was built by the NovaAssist team.",
    "who_do_you_work_for": "I work for NovaAssist customers - like you!",
    "what_can_i_ask_you": "You can ask me about banking, cards, travel, your car, your home and more.",
    "what_are_your_hobbies": "I enjoy routing messages to the right team.",
    "do_you_have_pets": "No pets, but I like hearing about yours.",
    "meaning_of_life": "42 - at least according to Douglas Adams.",
    "fun_fact": "Fun fact: honey never spoils.",
}


def reflect(fragment: str) -> str:
    return " ".join(REFLECTIONS.get(w.lower(), w) for w in fragment.split())


def reply(text: str, rng=random.Random(0)):
    t = text.strip()
    for pattern, templates in COMPILED:
        m = pattern.search(t)
        if m:
            groups = [reflect(g) for g in m.groups() if g]
            return rng.choice(templates).format(*groups) if groups else rng.choice(templates)
    return None


def reply_for_intent(intent: str) -> str:
    return INTENT_REPLIES.get(intent, "Nice talking to you! How can I help?")


if __name__ == "__main__":
    for s in ["hello there", "thank you so much", "i feel lost with my account",
              "are you a bot?", "my card was declined"]:
        print(f"{s!r:35} -> {reply(s)}")
