
if [ -d /home/datasets/mayilong/PycharmProjects/p55/data/rgb ]; then
        echo "Need to clean old file before extract"
        rm -rf /home/datasets/mayilong/PycharmProjects/p55/data/rgb/*
        echo "Clean Finish"
fi
echo ""

echo "save_root= /home/datasets/mayilong/PycharmProjects/p55/data/rgb"
echo "label_root= /home/datasets/mayilong/PycharmProjects/p55/data/split_data"
echo "err_file= /home/datasets/mayilong/PycharmProjects/p55/data/plan2/err.txt"

echo ""

echo "Extract Start"

python ./extract_rgb.py \
--label_root='/home/datasets/mayilong/PycharmProjects/p55/data/split_data' \
--video2path='/home/datasets/mayilong/PycharmProjects/p55/resource/train_val_video2path.pkl' \
--save_root='/home/datasets/mayilong/PycharmProjects/p55/data/rgb' \
--n_frame=16 \
--err_file='/home/datasets/mayilong/PycharmProjects/p55/data/err_record.txt' \

echo "Finish"



