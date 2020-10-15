from unittest import TestCase
from main import decay_pow2

class TestDecay(TestCase):
    def test_linear_decay_0_to_1(self):
        expected_0_to_1_in_10_steps=[
            0,.25,.5,.5,1,1 ,1,1 ,1 ,1 ,1
        ]
        for i in range(10):
            actual = decay_pow2(i, 11, 0, 1, 'linear')
            expected = expected_0_to_1_in_10_steps[i]
            self.assertEqual(expected,
                     actual,
                     msg="decay({},11,0,1,'linear') shouldBe {} but was {}"\
                             .format(i,expected, actual))

    def test_linear_decay_64_to_1(self):
        expected_64_to_1_in_10_steps=[
            64, 64, 64, 64, 32, 32, 32, 32, 16, 16, 1
        ]
        for i in range(10):
            actual = decay_pow2(i, 11, 64, 1, 'linear')
            expected = expected_64_to_1_in_10_steps[i]
            self.assertEqual(expected,
                     actual,
                     msg="decay({},11,64,1,'linear') shouldBe {} but was {}"\
                             .format(i,expected, actual))

    def test_exp_decay_64_to_1(self):
        expected_64_to_1_in_7_steps=[
            64, 32, 16, 8, 4, 2, 1
        ]
        for i in range(7):
            actual = decay_pow2(i, 7, 64, 1)
            expected = expected_64_to_1_in_7_steps[i]
            self.assertEqual(expected,
                     actual,
                     msg="decay({},7,64,1,) shouldBe {} but was {}"\
                             .format(i,expected, actual))

    def test_non_powers_of_2_should_fall_in_expected_range(self):

        for j in range(1,20):
            for i in range(0,j+1):
                from30= decay_pow2(i, j, 30, 1)
                from32= decay_pow2(i, j, 32, 1)
                from40= decay_pow2(i, j, 40, 1)
                from50= decay_pow2(i, j, 50, 1)
                from64= decay_pow2(i, j, 64, 1)
                self.assertTrue(from32 <= from40 <= from50 <= from64, "failed at step {} of {}"\
                                .format(i,j))
