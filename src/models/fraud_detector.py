def is_fraudulent(transaction):
    """Basic fraud detection logic (replace with ML)"""
    if transaction["amount"] > 900:  # Simple rule-based check
        return True
    return False
