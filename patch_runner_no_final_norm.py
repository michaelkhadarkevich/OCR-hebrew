from pathlib import Path
p = Path('/home/maxim/HTR-VT/run_omer_ariel_cpu.py')
text = p.read_text(encoding='utf-8')
if 'import types' not in text:
    text = text.replace('import csv\n', 'import csv\nimport types\n')
if "parser.add_argument('--no-final-logit-norm'" not in text:
    text = text.replace("parser.add_argument('--blank-bias', type=float, default=None)", "parser.add_argument('--blank-bias', type=float, default=None)\n    parser.add_argument('--no-final-logit-norm', action='store_true')")
helper = '''\n\ndef patch_forward_no_final_norm(model):\n    def forward(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False):\n        x = self.layer_norm(x)\n        x = self.patch_embed(x)\n        b, c, w, h = x.shape\n        x = x.view(b, c, -1).permute(0, 2, 1)\n        if use_masking:\n            x = self.random_masking(x, mask_ratio, max_span_length)\n        x = x + self.pos_embed\n        for blk in self.blocks:\n            x = blk(x)\n        x = self.norm(x)\n        return self.head(x)\n    model.forward = types.MethodType(forward, model)\n'''
if 'def patch_forward_no_final_norm' not in text:
    text = text.replace('\n\ndef collate(batch):', helper + '\n\ndef collate(batch):')
needle = "model = HTR_VT.create_model(nb_cls=nb_cls, img_size=[64, 512]).to(device)\n    if args.blank_bias is not None:"
if needle in text and 'if args.no_final_logit_norm' not in text:
    text = text.replace(needle, "model = HTR_VT.create_model(nb_cls=nb_cls, img_size=[64, 512]).to(device)\n    if args.no_final_logit_norm:\n        patch_forward_no_final_norm(model)\n        print('no_final_logit_norm=True')\n    if args.blank_bias is not None:")
p.write_text(text, encoding='utf-8')
print('patched runner')
