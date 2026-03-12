import unittest

import torch

from vqvae import VQVAE


class TestVQVAEInstantiation(unittest.TestCase):
    def test_default_instantiation(self):
        """VQVAE can be created with default hyperparameters."""
        model = VQVAE()
        self.assertIsInstance(model, VQVAE)

    def test_custom_instantiation(self):
        """VQVAE can be created with custom hyperparameters."""
        model = VQVAE(
            in_channel=3,
            channel=64,
            n_res_block=2,
            n_res_channel=16,
            embed_dim=32,
            n_embed=256,
            decay=0.99,
        )
        self.assertIsInstance(model, VQVAE)


class TestVQVAEOnCPU(unittest.TestCase):
    def setUp(self):
        self.model = VQVAE().eval()
        # Small 64x64 image so the test is fast
        self.batch = torch.randn(2, 3, 64, 64)

    def test_forward_output_shapes(self):
        """Forward pass returns a reconstruction and a scalar latent loss."""
        with torch.no_grad():
            recon, diff = self.model(self.batch)
        self.assertEqual(recon.shape, self.batch.shape,
                         "Reconstruction must have the same shape as input")
        self.assertEqual(diff.shape, torch.Size([1]),
                         "Latent loss must be a scalar tensor of shape [1]")

    def test_encode_output_shapes(self):
        """Encode returns two quantised codes and correct index shapes."""
        with torch.no_grad():
            quant_t, quant_b, diff, id_t, id_b = self.model.encode(self.batch)

        # Top-level feature map is downsampled by stride=4 then stride=2 → 8×
        expected_t_h = self.batch.shape[2] // 8
        expected_t_w = self.batch.shape[3] // 8
        self.assertEqual(quant_t.shape[2], expected_t_h)
        self.assertEqual(quant_t.shape[3], expected_t_w)

        # Bottom-level feature map is downsampled by stride=4 → 4×
        expected_b_h = self.batch.shape[2] // 4
        expected_b_w = self.batch.shape[3] // 4
        self.assertEqual(quant_b.shape[2], expected_b_h)
        self.assertEqual(quant_b.shape[3], expected_b_w)

    def test_decode_output_shape(self):
        """Decode from quantised codes reconstructs the original spatial size."""
        with torch.no_grad():
            quant_t, quant_b, _, _, _ = self.model.encode(self.batch)
            recon = self.model.decode(quant_t, quant_b)
        self.assertEqual(recon.shape, self.batch.shape)

    def test_decode_code_output_shape(self):
        """decode_code from discrete indices reproduces the original spatial size."""
        with torch.no_grad():
            _, _, _, id_t, id_b = self.model.encode(self.batch)
            recon = self.model.decode_code(id_t, id_b)
        self.assertEqual(recon.shape, self.batch.shape)

    def test_forward_train_mode(self):
        """Forward pass in train mode does not raise errors."""
        model = VQVAE().train()
        recon, diff = model(self.batch)
        self.assertEqual(recon.shape, self.batch.shape)


@unittest.skipUnless(torch.cuda.is_available(), "No CUDA GPU available – skipping GPU tests")
class TestVQVAEOnGPU(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.model = VQVAE().to(self.device).eval()
        self.batch = torch.randn(2, 3, 64, 64, device=self.device)

    def test_model_parameters_on_gpu(self):
        """All model parameters and buffers must reside on the GPU."""
        for name, param in self.model.named_parameters():
            self.assertTrue(param.is_cuda,
                            f"Parameter '{name}' is not on GPU")
        for name, buf in self.model.named_buffers():
            self.assertTrue(buf.is_cuda,
                            f"Buffer '{name}' is not on GPU")

    def test_forward_on_gpu(self):
        """Forward pass on GPU returns correct shapes and GPU tensors."""
        with torch.no_grad():
            recon, diff = self.model(self.batch)
        self.assertTrue(recon.is_cuda, "Reconstruction tensor must be on GPU")
        self.assertTrue(diff.is_cuda, "Latent loss tensor must be on GPU")
        self.assertEqual(recon.shape, self.batch.shape)
        self.assertEqual(diff.shape, torch.Size([1]))

    def test_encode_on_gpu(self):
        """Encode on GPU returns GPU tensors with correct shapes."""
        with torch.no_grad():
            quant_t, quant_b, diff, id_t, id_b = self.model.encode(self.batch)

        for tensor, name in [(quant_t, "quant_t"), (quant_b, "quant_b"),
                              (diff, "diff"), (id_t, "id_t"), (id_b, "id_b")]:
            self.assertTrue(tensor.is_cuda, f"'{name}' is not on GPU")

        expected_t_h = self.batch.shape[2] // 8
        expected_t_w = self.batch.shape[3] // 8
        self.assertEqual(quant_t.shape[2], expected_t_h)
        self.assertEqual(quant_t.shape[3], expected_t_w)

    def test_decode_on_gpu(self):
        """Decode on GPU returns a GPU tensor with the original spatial size."""
        with torch.no_grad():
            quant_t, quant_b, _, _, _ = self.model.encode(self.batch)
            recon = self.model.decode(quant_t, quant_b)
        self.assertTrue(recon.is_cuda, "Decoded tensor must be on GPU")
        self.assertEqual(recon.shape, self.batch.shape)

    def test_decode_code_on_gpu(self):
        """decode_code on GPU returns a GPU tensor with the original spatial size."""
        with torch.no_grad():
            _, _, _, id_t, id_b = self.model.encode(self.batch)
            recon = self.model.decode_code(id_t, id_b)
        self.assertTrue(recon.is_cuda, "Decoded tensor must be on GPU")
        self.assertEqual(recon.shape, self.batch.shape)

    def test_gpu_cpu_output_consistency(self):
        """CPU and GPU forward passes should produce numerically close results."""
        model_cpu = VQVAE().eval()
        model_cpu.load_state_dict(self.model.state_dict())

        batch_cpu = self.batch.cpu()
        with torch.no_grad():
            recon_gpu, _ = self.model(self.batch)
            recon_cpu, _ = model_cpu(batch_cpu)

        torch.testing.assert_close(
            recon_gpu.cpu(), recon_cpu,
            rtol=1e-4, atol=1e-4,
            msg="GPU and CPU outputs differ beyond tolerance",
        )


if __name__ == "__main__":
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU             : none – GPU tests will be skipped")
    print()
    unittest.main(verbosity=2)
