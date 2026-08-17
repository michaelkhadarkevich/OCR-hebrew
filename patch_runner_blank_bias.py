from pathlib import Path
p=Path('/home/maxim/HTR-VT/run_omer_ariel_cpu.py')
text=p.read_text(encoding='utf-8')
text=text.replace("parser.add_argument('--out-dir', default='output/omer_ariel_cpu_50')", "parser.add_argument('--out-dir', default='output/omer_ariel_cpu_50')\n    parser.add_argument('--blank-bias', type=float, default=None)")
text=text.replace("model = HTR_VT.create_model(nb_cls=nb_cls, img_size=[64, 512]).to(device)", "model = HTR_VT.create_model(nb_cls=nb_cls, img_size=[64, 512]).to(device)\n    if args.blank_bias is not None:\n        with torch.no_grad():\n            model.head.bias.zero_()\n            model.head.bias[0] = args.blank_bias\n        print('blank_bias={}'.format(args.blank_bias))")
p.write_text(text, encoding='utf-8')
print('patched blank-bias option')
