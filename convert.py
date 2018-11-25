import tensorflow as tf
import utils.dataset_reader
import numpy as np
from nn.dqn import DQN
from nn.generative import Generative
from PIL import Image
import collections
import os

_NEW_ = True
root_path = 'data'


def show_image(arr):
    formatted = (arr * 255).astype('uint8')
    img = Image.fromarray(formatted)
    img.show()

def show_record(data, type, frame):
    if type == 'query':
        frame = data.query.context.frames[0][0]
    else:
        frame = data.target
    show_image(frame)

# print(dataset.output_types)  # ==> "tf.float32"
# print(dataset.output_shapes)  # ==> "(10,)"
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])
def read(data):
    """Reads batch_size (query, target) pairs."""
    frames, cameras = data
    print("read cam",cameras.shape)
    try:
        frames = np.squeeze(frames, axis=1)
        cameras = np.squeeze(frames, axis=1)
    except ValueError:
        pass
    context_frames = frames[:, :-1]
    context_cameras = cameras[:, :-1]
    target = frames[:, -1]
    query_camera = cameras[:, -1]
    context = Context(cameras=context_cameras, frames=context_frames)
    query = Query(context=context, query_camera=query_camera)
    return TaskData(query=query, target=target)

def convert_dataset(dataset, root):
    for mode in ["train", "test"]:
        print("Mode:", mode)
        dataset = utils.dataset_reader.make_dataset(dataset='shepard_metzler_5_parts', mode=mode, root=root_path, load_all=True)
        # dataset = dataset.batch(12)
        iterator = dataset.make_one_shot_iterator()
        next_elem = iterator.get_next()
        writer = utils.dataset_reader.DatasetWriter(dataset='shepard_metzler_5_parts', mode=mode, root=os.path.join(root_path, 'general'))
        count = 0
        with tf.Session() as sess:
            records = []
            try:
                while True:
                    data = sess.run(next_elem)
                    records.append(data)
                    count += 1
                    if count % 1000 == 0:
                        writer.save_multiple(records)
                        records = []
            except tf.errors.OutOfRangeError:
                if count % 1000 != 0:
                    writer.save_multiple(records)
            print("total:", count)
            with open(writer.meta_file, 'w') as f:
                f.write(str(count))
            

if __name__ == "__main__":
    convert_dataset(dataset='shepard_metzler_5_parts', root=root_path)
