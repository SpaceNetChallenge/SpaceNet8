class Callbacks:
    def __init__(self):
        self._callbacks = {
            'on_train_start': [],
            'on_train_end': [],
            'on_train_batch_start': [],
            'on_train_batch_end': [],
            'on_train_epoch_start': [],
            'on_train_epoch_end': [],
            'on_val_start': [],
            'on_val_batch_start': [],
            'on_val_image_end': [],
            'on_val_batch_end': [],
            'on_val_end': [],
            'on_fit_epoch_end': [],  # fit = train + val
            'on_model_save': [],
        }
        self.stop_training = False

    def register_action(self, hook, name='', callback=None):
        self._callbacks[hook].append({'name': name, 'callback': callback})

    def get_registered_actions(self, hook=None):
        if hook:
            return self._callbacks[hook]
        else:
            return self._callbacks

    def run(self, hook, *args, **kwargs):
        for logger in self._callbacks[hook]:
            logger['callback'](*args, **kwargs)

