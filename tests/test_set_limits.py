import unittest

from pyscipopt import Model

from gyozas.instances import InstanceGenerator
from gyozas.instances.modifiers import SetLimits


class DummyInstanceGenerator(InstanceGenerator):
    def __init__(self):
        super().__init__()
        self._seed = None

    def seed(self, seed):
        self._seed = seed

    def generate_instance(self, *args, **kwargs):
        return Model()

    def __next__(self):
        return Model()


class TestSetLimits(unittest.TestCase):
    def setUp(self):
        self.generator = DummyInstanceGenerator()

    def test_all_limits_set(self):
        model = next(SetLimits(self.generator, time=5.0, nodes=100, gap=0.01))
        self.assertEqual(model.getParam("limits/time"), 5.0)
        self.assertEqual(model.getParam("limits/nodes"), 100)
        self.assertEqual(model.getParam("limits/gap"), 0.01)

    def test_only_provided_limits_are_set(self):
        default_time = Model().getParam("limits/time")
        model = next(SetLimits(self.generator, nodes=42))
        self.assertEqual(model.getParam("limits/nodes"), 42)
        # Limits left as None keep SCIP's default.
        self.assertEqual(model.getParam("limits/time"), default_time)

    def test_no_limits_leaves_defaults(self):
        reference = Model()
        model = next(SetLimits(self.generator))
        self.assertEqual(model.getParam("limits/time"), reference.getParam("limits/time"))
        self.assertEqual(model.getParam("limits/nodes"), reference.getParam("limits/nodes"))
        self.assertEqual(model.getParam("limits/gap"), reference.getParam("limits/gap"))

    def test_parameters_set_alongside_limits(self):
        model = next(SetLimits(self.generator, nodes=10, parameters={"display/verblevel": 0}))
        self.assertEqual(model.getParam("limits/nodes"), 10)
        self.assertEqual(model.getParam("display/verblevel"), 0)

    def test_explicit_limit_overrides_parameters(self):
        model = next(SetLimits(self.generator, time=5.0, parameters={"limits/time": 99.0}))
        self.assertEqual(model.getParam("limits/time"), 5.0)

    def test_parameters_argument_not_mutated(self):
        params = {"limits/time": 99.0}
        next(SetLimits(self.generator, time=5.0, parameters=params))
        self.assertEqual(params, {"limits/time": 99.0})

    def test_generate_instance_applies_limits(self):
        model = SetLimits(self.generator, time=3.0).generate_instance()
        self.assertEqual(model.getParam("limits/time"), 3.0)

    def test_seed_forwarded_to_inner_generator(self):
        SetLimits(self.generator, nodes=1).seed(42)
        self.assertEqual(self.generator._seed, 42)


if __name__ == "__main__":
    unittest.main()
