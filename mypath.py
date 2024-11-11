class Pather:
    """
    This class provides the root directory of different datasets on different platforms.
    """

    @staticmethod
    def db_root_dir(platform: str = None, dataset: str = None) -> str:
        """
            This class provides the root directory of different datasets on different platforms.

            Args:
                platform (str): The platform name, such as 'Windows', 'Linux', 'Remote'.
                dataset (str): The dataset name, such as 'Pascal', 'Crack500'.

            Returns:
                str: The root directory of the dataset on the platform.
            """
        platform = platform.lower()
        dataset = dataset.lower()

        # 此处可以根据平台和数据集的不同，返回不同的数据集的根目录，修改为自己的路径
        if platform == 'windows':
            if dataset == 'crack500':
                return 'D:\\Code&Project\\Pytorch\\Datasets\\Crack500_ori'  # for windows dir
            else:
                print('Dataset {} is not available'.format(dataset))
                raise NotImplementedError
        elif platform == 'remote':
            return '/home/featurize/work/dataset/Crack500_ori'
        elif platform == 'linux':
            if dataset == 'pascal':
                return '\home\wjh\dataset\VOC'  # folder that contains VOCdevkit/
            elif dataset == 'crack500':
                return '/home/wjh/dataset/Crack500_ori'  # for linux dir
            else:
                print('Dataset {} is not available'.format(dataset))
                raise NotImplementedError
        else:
            print('Platform {} is not available'.format(platform))
            raise NotImplementedError


if __name__ == '__main__':
    # platform_name = 'WIndows'
    # dataset_name = 'CRack500'
    # get_db_dir = Pather(None).db_root_dir
    # db_dir = get_db_dir(platform_name, dataset_name)
    # print("The root directory of {} dataset on {} platform is: {}".format(dataset_name, platform_name, db_dir))
    #
    # import os
    #
    # print(os.path.exists(db_dir))
    from tqdm import tqdm
    from time import sleep
    tbar = tqdm(range(100), bar_format='{l_bar}{n_fmt}/{total_fmt} [TimeUsed:{elapsed} TimeRemaining:{remaining}, Speed:{rate}/{postfix}, loss:{loss:.3f}]',
                total=100 * 4, unit='images', colour='blue', ncols=60)
    for i in tbar:
        sleep(0.3)
        tbar.set_description('Epoch %d' % i)
        tbar.set_postfix(rate=4*tbar.rate, loss=i)
        tbar.update(4)
        tbar.refresh()
