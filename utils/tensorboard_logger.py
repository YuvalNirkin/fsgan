# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """ Resets the statistics of previous values. """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Add new value.

        Args:
            val (float): Value to add
            n (int): Count the value n times. Default is 1
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TensorBoardLogger(SummaryWriter):
    """ Writes entries directly to event files in the logdir to be consumed by TensorBoard.

    The logger keeps track of scalar values, allowing to easily log either the last value or the average value.

    Args:
        log_dir (str): The directory in which the log files will be written to
    """

    def __init__(self, log_dir=None):
        super(TensorBoardLogger, self).__init__(log_dir)
        self.__tb_logger = SummaryWriter(log_dir) if log_dir is not None else None
        self.log_dict = {}

    def reset(self, prefix=None):
        """ Resets all saved scalars and description prefix.

        Args:
            prefix (str, optional): The logger's prefix description used when printing the logger status
        """
        self.prefix = prefix
        self.log_dict.clear()

    def update(self, category='losses', **kwargs):
        """ Add named scalar values to the logger. If a scalar with the same name already exists, the new value will
        be associated with it.

        Args:
            category (str): The scalar category that will be concatenated with the main tag
            **kwargs: Named scalar values to be added to the logger.
        """
        if category not in self.log_dict:
            self.log_dict[category] = {}
        category_dict = self.log_dict[category]
        for key, val in kwargs.items():
            if key not in category_dict:
                category_dict[key] = AverageMeter()
            category_dict[key].update(val)

    def log_scalars_val(self, main_tag, global_step=None):
        """ Log the last value of all scalars.

        Args:
            main_tag (str): The parent name for the tags
            global_step (int, optional): Global step value to record
        """
        if self.__tb_logger is not None:
            for category, category_dict in self.log_dict.items():
                val_dict = {k: v.val for k, v in category_dict.items()}
                self.__tb_logger.add_scalars(main_tag + '/' + category, val_dict, global_step)

    def log_scalars_avg(self, main_tag, global_step=None):
        """ Log the average value of all scalars.

        Args:
            main_tag (str): The parent name for the tags
            global_step (int, optional): Global step value to record
        """
        if self.__tb_logger is not None:
            for category, category_dict in self.log_dict.items():
                val_dict = {k: v.avg for k, v in category_dict.items()}
                self.__tb_logger.add_scalars(main_tag + '/' + category, val_dict, global_step)

    def log_image(self, tag, img_tensor, global_step=None):
        """ Add an image tensor to the log.

        Args:
            tag (str): Name identifier for the image
            img_tensor (torch.Tensor): The image tensor to log
            global_step (int, optional): Global step value to record
        """
        if self.__tb_logger is not None:
            self.__tb_logger.add_image(tag, img_tensor, global_step)

    def __str__(self):
        desc = '' if self.prefix is None else self.prefix
        for category, category_dict in self.log_dict.items():
            desc += '{}: ['.format(category)
            for key, log in category_dict.items():
                desc += '{}: {:.4f} ({:.4f}); '.format(key, log.val, log.avg)
            desc += '] '

        return desc
