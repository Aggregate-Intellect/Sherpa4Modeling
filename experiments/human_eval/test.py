from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    # Remove spaces from the input string
    paren_string = paren_string.replace(" ", "")
    
    groups = []
    balance = 0
    start_index = 0
    
    for i, char in enumerate(paren_string):
        if char == '(':
            if balance == 0:
                start_index = i  # Mark the start of a new group
            balance += 1
        elif char == ')':
            balance -= 1
            if balance == 0:
                # We found a complete group
                groups.append(paren_string[start_index:i + 1])
    
    return groups

def check(candidate):
    # Test with multiple groups of parentheses
    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
    
    # Test with a single group of parentheses
    assert candidate('()') == ['()']
    
    # Test with nested parentheses (should ignore nesting)
    assert candidate('((()))') == ['((()))']
    
    # Test with spaces and multiple groups
    assert candidate(' ( )   ( ( ) )   ( ( ) ( ) ) ') == ['()', '(())', '(()())']
    
    # Test with no parentheses
    assert candidate('') == []
check(separate_paren_groups)