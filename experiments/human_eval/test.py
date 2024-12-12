from typing import List
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    # Sort the numbers
    numbers.sort()
    
    # Check adjacent elements
    for i in range(len(numbers) - 1):
        if abs(numbers[i] - numbers[i + 1]) <= threshold:
            return True
            
    return False

def check(candidate):
    # No elements
    assert candidate([], 0.5) == False
    # Single element
    assert candidate([1.0], 0.5) == False
    # All elements far apart
    assert candidate([1.0, 2.0, 3.0], 0.5) == False
    # At least one pair close
    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
    # Threshold is zero (checking for duplicates)
    assert candidate([1.0, 1.0, 2.0], 0.0) == True

check(has_close_elements)