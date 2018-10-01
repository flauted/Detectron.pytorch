for i in $( ls Outputs/e2e_mask_rcnn_R-101-FPN_2x/Sep22-15-48-46_aurora_step/ckpt ); do
	fldr="Outputs/e2e_mask_rcnn_R-101-FPN_2x/Sep22-15-48-46_aurora_step/test/$i"
	python tools/test_net.py --dataset isic2018 --cfg configs/baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml --load_ckpt Outputs/e2e_mask_rcnn_R-101-FPN_2x/Sep22-15-48-46_aurora_step/ckpt/$i --output_dir ${fldr}
done
