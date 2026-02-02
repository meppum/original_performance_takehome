import unittest


class PerfTakehomeIntegrationTests(unittest.TestCase):
    def test_kernelbuilder_build_kernel_smoke(self):
        from perf_takehome import KernelBuilder
        from problem import SCRATCH_SIZE, VLEN

        forest_height = 3
        n_nodes = (1 << (forest_height + 1)) - 1
        batch_size = VLEN  # smallest supported vector batch
        rounds = 4

        kb = KernelBuilder()
        kb.build_kernel(forest_height, n_nodes, batch_size, rounds)

        self.assertGreater(len(kb.instrs), 0)
        self.assertGreater(kb.scratch_ptr, 0)
        self.assertLessEqual(kb.scratch_ptr, SCRATCH_SIZE)


if __name__ == "__main__":
    unittest.main()

