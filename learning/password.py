# Password strength
# less than 8 symbols - return Too Weak
# only numbers, upper, lower - Weak
# 2 of 3 lists (numbers, upper, lower) - Good
# 3 of 3 lists (numbers, upper, lower) - Very Good
# n=input('Password: ')
import string


def password_check(value: str) -> str:
    digits = string.digits
    lowers = string.ascii_lowercase
    uppers = string.ascii_uppercase
    if len(value) < 8:
        return 'Too Weak'
    if all(e in digits for e in value) or all(e in lowers for e in value) or all(e in uppers for e in value):
        return 'Weak'
    if any(e in digits for e in value) and any(e in lowers for e in value) and any(e in uppers for e in value):
        return 'Very Good'
    # if (any(e in digits for e in value) and any(e in lowers for e in value)) or (
    #         any(e in digits for e in value) and any(e in uppers for e in value)) or (
    #         any(e in lowers for e in value) and any(e in uppers for e in value)):
    return 'Good'
    #


# password_check(n)
if __name__ == '__main__':
    assert password_check('') == 'Too Weak'
    assert password_check('1234567') == 'Too Weak'
    assert password_check('asdfghj') == 'Too Weak'
    assert password_check('ASDFVCD') == 'Too Weak'
    assert password_check('12345678131') == 'Weak'
    assert password_check('asdfghjkadasd') == 'Weak'
    assert password_check('ASDFGHJKASFAFDA') == 'Weak'
    assert password_check('1234fasawq') == 'Good'
    assert password_check('1234FAST') == 'Good'
    assert password_check('awqeDAQW') == 'Good'
    assert password_check('123FASTkafj') == 'Very Good'
