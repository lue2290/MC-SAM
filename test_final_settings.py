"""CPU regression tests of final-setting modules without loading SAM weights."""
import ast
from pathlib import Path
import unittest
import torch
from torch import nn
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'segment_anything/modeling/mcsam_integrated.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
names = {'SinkhornProjection', 'HyperCondModule', 'RankDiceRMAModule',
         'LowRankLinear', 'DepthwiseSeparableConv',
         'CrossModalStablePromptGenerator', 'MMSAM_Integrated'}
namespace = {'torch': torch, 'nn': nn, 'F': F}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names], type_ignores=[]), str(SOURCE), 'exec'), namespace)

class FinalSettingsTests(unittest.TestCase):
    def test_gram_and_column_post_mapping(self):
        torch.manual_seed(7)
        m = namespace['CrossModalStablePromptGenerator'](
            text_dim=6, vision_dim=7, prompt_dim=8, projection_rank=4
        )
        t, v = torch.randn(3,6), torch.randn(3,7)
        p = torch.stack((m.W_text(t),m.W_vision(v)),dim=1)
        h = m.compute_H_pre(p)
        u = F.normalize(p,dim=-1)
        expected = namespace['SinkhornProjection'](iters=5)(-(u@u.transpose(-1,-2)))
        torch.testing.assert_close(h,expected)
        torch.testing.assert_close(h.sum(-1),torch.ones(3,2),atol=1e-5,rtol=1e-5)
        mixed = h@p
        post = m.compute_H_post(mixed)
        torch.testing.assert_close(post.sum(-2),torch.ones(3,2))
        out = m(t,v)
        torch.testing.assert_close(out,post.transpose(-1,-2)@mixed)
        out.square().mean().backward()
        for layer in (m.W_text.down, m.W_text.up,
                      m.W_vision.down, m.W_vision.up, m.post_mapping):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(),0)

    def test_threshold_only(self):
        m=namespace['HyperCondModule']().eval()
        self.assertFalse(any('boundary' in n for n,_ in m.named_parameters()))
        a=m(.5,'cpu',3)
        self.assertEqual(a.shape,(3,128))
        torch.testing.assert_close(a[0],a[2])
        self.assertFalse(torch.allclose(a,m(.6,'cpu',3)))

    def test_rankdice_inference_only(self):
        m=namespace['RankDiceRMAModule']()
        self.assertEqual(sum(p.numel() for p in m.parameters()),0)
        z=torch.randn(2,1,4,4,requires_grad=True)
        self.assertIs(m(z,torch.zeros_like(z)),z)
        m.eval()
        torch.testing.assert_close(m(z,torch.zeros_like(z)),m(z,torch.ones_like(z)))
        self.assertTrue(set(m(z).unique().tolist()) <= {0.,1.})

    def test_integrated_training_and_dense_condition(self):
        class Encoder(nn.Module):
            def forward(self, image, features):
                return image.mean(1,keepdim=True).repeat(1,256,1,1)
        class Decoder(nn.Module):
            def forward(self, **kwargs):
                x=kwargs['image_embeddings']+kwargs['dense_prompt_embeddings']
                return x.mean(1,keepdim=True), None
        m=namespace['MMSAM_Integrated'].__new__(namespace['MMSAM_Integrated'])
        nn.Module.__init__(m)
        m.image_encoder=Encoder(); m.mask_decoder=Decoder()
        m.use_hypercond=True; m.use_rankdice=True
        m.hyper_cond=namespace['HyperCondModule']()
        m.prompt_generator=namespace['CrossModalStablePromptGenerator']()
        m.pseudo_mask_embed=nn.Conv2d(256,256,1)
        m.cond_channel_adapter=nn.Conv2d(128,256,1)
        m.cond_spatial_adapter=nn.Identity()
        m.pe_layer=lambda size: torch.zeros(256,*size)
        m.rankdice_module=namespace['RankDiceRMAModule']()
        m.register_buffer('training_step',torch.tensor(0))
        image=torch.randn(1,3,4,4,requires_grad=True)
        text=torch.randn(1,3,768); vision=torch.randn(1,768,4,4)
        def run(boundary):
            return m(image,text,vision,hyper_cond={'threshold':.5,'boundary_weight':boundary},return_logits=True)
        m.eval(); m.use_rankdice=False
        torch.testing.assert_close(run(.3)[0],run(1.)[0])
        m.train()
        # Training must never invoke the inference selector.
        m.use_rankdice=True
        m.rankdice_module._inference=lambda z: (_ for _ in ()).throw(AssertionError('selector called during training'))
        pred,loss,_=run(.3)
        self.assertEqual(loss.item(),0)
        pred.square().mean().backward()
        self.assertIsNotNone(m.hyper_cond.threshold_encoder[0].weight.grad)

    def test_training_objective_has_no_rank_loss(self):
        for file in [ROOT/'MCsam_train.py',SOURCE]:
            t=ast.parse(file.read_text(encoding='utf-8'))
            for n in ast.walk(t):
                if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='total_loss' for x in n.targets):
                    self.assertNotIn('rank_dice_loss',{x.id for x in ast.walk(n.value) if isinstance(x,ast.Name)})

if __name__=='__main__':unittest.main()
