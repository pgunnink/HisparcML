import tensorflow as tf

from keras.callbacks import TensorBoard
from tensorboard import summary as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2


class TB(TensorBoard):
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0

    def set_model(self, *args, **kwargs):
        super().set_model(*args, **kwargs)
        layout_summary = summary_lib.custom_scalar_pb(
            layout_pb2.Layout(category=[
                layout_pb2.Category(
                    title='compare losses',
                    chart=[
                        layout_pb2.Chart(
                            title='losses',
                            multiline=layout_pb2.MultilineChartContent(
                                tag=[r'.*loss.*'], )),
                    ])]))
        self.writer.add_summary(layout_summary)
    '''
    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

        super().on_batch_end(batch, logs)
    '''