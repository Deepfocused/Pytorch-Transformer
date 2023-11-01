from torch.utils.data import DataLoader

from core.utils.dataprocessing.Dataset import Dataset


def traindataloader(batch_size=256, pin_memory=True, num_workers=8):

    #num_workers = 0 if pin_memory else num_workers

    # shuffle 옵션 dataset에 list가 없어서 shuffle이 안된다.
    dataset = Dataset()
    dataloader = DataLoader(
        dataset.train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        num_workers=0)

    return dataloader, dataset


def validdataloader(batch_size=256, num_workers=8, pin_memory=True):

    #num_workers = 0 if pin_memory else num_workers

    dataset = Dataset()
    dataloader = DataLoader(
        dataset.valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        num_workers=0)

    return dataloader, dataset


def testdataloader(batch_size=256, pin_memory=True, num_workers=8):

    #num_workers = 0 if pin_memory else num_workers

    dataset = Dataset()
    dataloader = DataLoader(
        dataset.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        num_workers=0)

    return dataloader, dataset


# test
if __name__ == "__main__":

    train_dataloader, train_dataset = traindataloader(batch_size=256, pin_memory=False)
    for src, tgt in train_dataloader:
        print(src.shape)