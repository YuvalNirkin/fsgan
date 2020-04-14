import numpy as np
from visdom import Visdom


class VisdomLogger:
    def __init__(self, title, display_id=1, ncols=0, vis_server='http://localhost', vis_port=8097):
        self.title = title
        self.display_id = display_id
        self.ncols = ncols

        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(server=vis_server, port=vis_port)
            if not self.vis.check_connection():
                raise RuntimeError('''Visdom server is not running!
                    To start the Visdom server run: "python -m visdom.server".
                    You can also disable Visdom logging by setting the --visdom option to False.''')

    def display_images(self, labeled_images):
        if self.ncols > 0:
            ncols = min(self.ncols, len(labeled_images))
            h, w = next(iter(labeled_images.values())).shape[:2]
            table_css = """<style>
                    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                    </style>""" % (w, h)
            label_html = ''
            label_html_row = ''
            images = []
            idx = 0
            for label, image in labeled_images.items():
                image_numpy = tensor2im(image)
                label_html_row += '<td>%s</td>' % label
                images.append(image_numpy.transpose([2, 0, 1]))
                idx += 1
                if idx % ncols == 0:
                    label_html += '<tr>%s</tr>' % label_html_row
                    label_html_row = ''
            white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
            while idx % ncols != 0:
                images.append(white_image)
                label_html_row += '<td></td>'
                idx += 1
            if label_html_row != '':
                label_html += '<tr>%s</tr>' % label_html_row
            # pane col = image row
            self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                            padding=2, opts=dict(title=self.title + ' images'))
            label_html = '<table>%s</table>' % label_html
            self.vis.text(table_css + label_html, win=self.display_id + 2,
                          opts=dict(title=self.title + ' labels'))
        else:
            idx = 1
            for label, image in labeled_images.items():
                image_numpy = tensor2im(image)
                self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                               win=self.display_id + idx)
                idx += 1

    def plot_losses(self, epoch, counter_ratio, labeled_losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(labeled_losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([labeled_losses[k].item() for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.title + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)
