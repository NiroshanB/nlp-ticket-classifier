"""Generate a realistic synthetic support-ticket dataset.

The tickets are SYNTHETIC: template-based text with randomised slots
(product names, order numbers, amounts, dates). They are not real
customer messages. The fixed seed makes the dataset reproducible.
"""
import random

import pandas as pd

RANDOM_SEED = 42
N_PER_CATEGORY = 400

PRODUCTS = [
    "wireless headphones", "laptop stand", "mechanical keyboard", "USB-C hub",
    "smartwatch", "bluetooth speaker", "webcam", "monitor", "tablet",
    "noise cancelling earbuds", "gaming mouse", "external SSD",
]
NAMES = ["Alex", "Priya", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley"]

TEMPLATES = {
    "billing": [
        "I was charged twice for my {product} order, please refund the duplicate charge.",
        "My invoice shows the wrong amount, I was billed ${amount} instead of ${amount2}.",
        "Why was my card charged ${amount} when I cancelled before the renewal date?",
        "I need a copy of the receipt for order {order} for my expense report.",
        "The discount code did not apply at checkout and I was charged full price.",
        "My subscription renewed even though I turned off auto renewal last month.",
        "I was charged a late fee of ${amount} but I paid on time, please remove it.",
        "The refund for my returned {product} has not arrived after two weeks.",
        "Please explain the ${amount} charge on my statement dated {date}.",
        "My payment failed but the money was still taken from my account.",
    ],
    "technical": [
        "My {product} keeps disconnecting from bluetooth every few minutes.",
        "The {product} will not turn on even after charging overnight.",
        "After the latest update the app crashes every time I open the {product} settings.",
        "The {product} screen flickers and shows distorted colours when I plug it in.",
        "I cannot connect my {product} to wifi, it says the password is wrong but it is not.",
        "The {product} overheats after ten minutes of use and shuts down by itself.",
        "Sound only comes out of one side of my {product}, the other side is silent.",
        "My {product} freezes on the loading screen and I have to restart it.",
        "The firmware update for my {product} failed halfway and now it is stuck.",
        "The {product} battery drains from full to empty in under an hour.",
    ],
    "account": [
        "I forgot my password and the reset email never arrives in my inbox.",
        "Please help me change the email address linked to my account.",
        "My account was locked after too many login attempts, how do I unlock it?",
        "I want to delete my account and all of my personal data permanently.",
        "I cannot log in, it says my account does not exist but I registered last week.",
        "Please merge my two accounts, I accidentally created a second one.",
        "How do I change my username on my account profile?",
        "I need to update the phone number on my account for two factor authentication.",
        "My account shows orders I did not place, has it been hacked?",
        "Please close my account, I no longer want to use the service.",
    ],
    "shipping": [
        "My order {order} has been stuck in transit for five days with no update.",
        "The tracking page says my {product} was delivered but I never received it.",
        "Can I change the delivery address for order {order} before it ships?",
        "My package arrived damaged, the box was crushed and the {product} is dented.",
        "The delivery was supposed to arrive on {date} but it is still not here.",
        "I paid for express shipping but the order is arriving a week late.",
        "The courier left my package outside in the rain and the {product} is wet.",
        "Where is my order {order}? The tracking number does not work.",
        "I need my order delivered by {date} for a birthday gift, is that possible?",
        "The wrong item was shipped to me, I ordered a {product} and got something else.",
    ],
    "returns": [
        "I want to return my {product}, it does not fit and I have the receipt.",
        "How do I start a return for order {order}? The item arrived faulty.",
        "The return label you sent me does not work at the post office.",
        "I returned my {product} a week ago but have not received my refund yet.",
        "Can I exchange my {product} for a different colour instead of a refund?",
        "The return window says 30 days but I bought the {product} 35 days ago, can I still return it?",
        "I was told the return was approved but the pickup was never scheduled.",
        "Please send me a replacement for the broken {product} I returned.",
        "I returned the wrong item by mistake, can I get it back?",
        "The refund went to the wrong card, please send it to my new card instead.",
    ],
}


# Harder tickets whose wording overlaps other categories, so the task
# is not trivially separable by a single keyword.
AMBIGUOUS = {
    "billing": [
        "I need help with the charge for my delivery, the shipping fee on order {order} looks wrong.",
        "I paid for my {product} but the payment page shows the order as unpaid.",
        "The refund for the delivery I never received has not shown up on my card.",
        "My invoice for order {order} lists a delivery charge I did not agree to.",
    ],
    "technical": [
        "The app charged me twice when I tried to buy the {product}, and now the app will not open.",
        "My {product} arrived but the setup screen freezes and I cannot log in to my account.",
        "The {product} keeps crashing when I try to update my payment method.",
        "I cannot download the manual for my {product}, the link in my account page is broken.",
    ],
    "account": [
        "I cannot log in to check my order {order} and the tracking page asks me to pay again.",
        "Please update the address on my account, my {product} was sent to my old flat.",
        "I changed my password and now my order history for the {product} is empty.",
        "My account says my refund was sent but my card shows nothing, is my email correct?",
    ],
    "shipping": [
        "My order {order} was charged to my card but the tracking says it was returned to sender.",
        "The {product} I ordered was delivered to the wrong address on my account.",
        "I paid extra for fast delivery but the package arrived after my refund was processed.",
        "The courier could not deliver my {product} because the payment for shipping failed.",
    ],
    "returns": [
        "I returned my {product} and the refund went through, but the replacement order was never shipped.",
        "The return pickup for my {product} was missed and now my account shows a late fee.",
        "I want to exchange my {product} but the website charged me again instead of swapping it.",
        "My returned {product} arrived at your warehouse but my card has not been refunded yet.",
    ],
}
N_AMBIGUOUS_PER_CATEGORY = 40


def build_text(template: str, rng: random.Random) -> str:
    return template.format(
        product=rng.choice(PRODUCTS),
        order=f"ORD-{rng.randint(10000, 99999)}",
        amount=rng.randint(15, 250),
        amount2=rng.randint(15, 250),
        date=f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        name=rng.choice(NAMES),
    )


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    rows = []
    for category, templates in TEMPLATES.items():
        for _ in range(N_PER_CATEGORY):
            template = rng.choice(templates)
            rows.append({"text": build_text(template, rng), "category": category})
    rng.shuffle(rows)
    # Ambiguous tickets with cross-category wording (harder examples).
    for category, templates in AMBIGUOUS.items():
        for _ in range(N_AMBIGUOUS_PER_CATEGORY):
            template = rng.choice(templates)
            rows.append({"text": build_text(template, rng), "category": category})
    rng.shuffle(rows)
    # Real support queues are messy: drop a random word from 10% of
    # tickets, and mislabel 3% of them. This keeps the task realistic
    # instead of trivially separable by template fingerprints.
    for row in rows:
        if rng.random() < 0.10:
            words = row["text"].split()
            if len(words) > 5:
                words.pop(rng.randrange(len(words)))
            row["text"] = " ".join(words)
    n_noisy = int(0.03 * len(rows))
    for row in rng.sample(rows, n_noisy):
        others = [c for c in TEMPLATES if c != row["category"]]
        row["category"] = rng.choice(others)
    df = pd.DataFrame(rows, columns=["text", "category"])
    df.to_csv("data/tickets.csv", index=False)
    print(f"Wrote {len(df)} tickets to data/tickets.csv")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
