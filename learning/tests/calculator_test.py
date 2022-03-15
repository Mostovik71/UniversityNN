from unittest import TestCase, main
from learning.calculator import calculator


class CalculatorTest(TestCase):
    def test_plus(self):
        self.assertEqual(calculator('2+2'), 4)

    def test_minus(self):
        self.assertEqual(calculator('4-3'), 1)

    def test_multi(self):
        self.assertEqual(calculator('5*10'), 50)

    def test_multi(self):
        self.assertEqual(calculator('5*10'), 50)

    def test_divide(self):
        self.assertEqual(calculator('15/5'), 3)

    def test_no_signs(self):
        with self.assertRaises(ValueError) as e:
            calculator('abcde')
        self.assertEqual('Выражение должно содержать хотя бы один знак (+-*/)', e.exception.args[0])

    def test_two_signs(self):
        with self.assertRaises(ValueError) as e:
            calculator('2+2+2')
        self.assertEqual('Выражение должно содержать 2 целых числа и один знак', e.exception.args[0])

    def test_two_signs(self):
        with self.assertRaises(ValueError) as e:
            calculator('2+3*10')
        self.assertEqual('Выражение должно содержать 2 целых числа и один знак', e.exception.args[0])

    def test_int(self):
        with self.assertRaises(ValueError) as e:
            calculator('2.5+3')
        self.assertEqual('Выражение должно содержать 2 целых числа и один знак', e.exception.args[0])


if __name__ == '__main__':
    main()
