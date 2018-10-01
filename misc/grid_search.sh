cd "mask-rcnn.pytorch"

PT_MODELS_DIR="data/pretrained_model"
RUN_NAME="grid_search"
OUTPUT_CACHE="/run/media/dylan/My Passport/maskrcnn_isic_models"

for pt_model_file in $( ls ${PT_MODELS_DIR} | grep mask_rcnn ); do
	arch=${pt_model_file%.pkl}
	python tools/train_net_step.py --dataset isic2018 --cfg configs/baselines/${arch}.yaml --load_detectron ${PT_MODELS_DIR}/${pt_model_file} --xfer --bs 1 --run_name ${RUN_NAME} --set TRAIN.SNAPSHOT_ITERS 5000 SOLVER.MAX_ITER 2200 

	ckpts_dir="Outputs/${arch}/${RUN_NAME}_step/ckpt"
	for ckpt in $( ls $ckpts_dir ); do
		test_out="Outputs/${arch}/${RUN_NAME}_step/test/${ckpt}"
		python tools/test_net.py --dataset isic2018 --cfg configs/baselines/${arch}.yaml --load_ckpt ${ckpts_dir}/${ckpt} --output_dir ${test_out}
	done
	mv "Outputs/${arch}" "${OUTPUT_CACHE}/${arch}"
	mkdir -p "${PT_MODELS_DIR}/complete"
	mv "${PT_MODELS_DIR}/${pt_model_file}" "${PT_MODELS_DIR}/complete"
done
