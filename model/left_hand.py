"""
Left hand
____________

This module contains classes and functions for handling left hand inference on the FreiHand dataset and external images.

Classes:
- FreiHANDLeftHandDataset: A specialized dataset class for left hand data from the FreiHand dataset.
- FreiHANDLeftHandDataLoader: Implements the DataLoader for the FreiHand left hand dataset.
- AlexNet: AlexNet model architecture for hand pose estimation.
- LeftHandCollectionHandInference: A specialized class for managing a collection
                                   of both left and right hands and performing inference on hand pose estimation results.
- ExternalLeftHand: Performs inference on an external hand image (left or right) using a trained model.
"""

from typing import Dict, Any, Optional, Union, Iterable, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t

from io_ import log, load_model, load_image, load_external_image
from model.dataset import FreiHANDDataset, FreiHANDDataLoader
from model.hand import HandCollection, Hand
from model.inference import HandCollectionInference, InferenceHand, ExternalHand
from model.network import HandPoseEstimationUNet
from settings import FREIHAND_INFO


class FreiHANDLeftHandDataset(FreiHANDDataset):
    """
    A specialized dataset class for left hand data from the FreiHand dataset.

    This class extends the FreiHANDDataset class to provide functionality for handling left hand data.
    """

    def __init__(self, set_type: str):
        """
        Initialize the FreiHANDLeftHandDataset instance.

        :param set_type: name of set (train, val or test)
        """

        super().__init__(set_type)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Get a data sample from the dataset.

        :param idx: Index of the data sample.
        :return: dictionary containing the data sample.
        """

        item = super().__getitem__(idx)

        right_hand = idx % 2 == 0

        X = item["image"]
        X = nn.functional.interpolate(X.unsqueeze(0), size=(227, 227), mode='bilinear', align_corners=False).squeeze()
        if not right_hand:
            X = torch.flip(X, [2])

        label = 0 if right_hand else 1
        y = torch.tensor(data=label)

        return {
            "image": X,
            "left": y,
            "image_name": item["image_name"],
        }


class FreiHANDLeftHandDataLoader(FreiHANDDataLoader):
    """
    This class implements the DataLoader for the FreiHand left hand dataset.
    """

    # CONSTRUCTOR OVERRIDE

    def __init__(self, dataset: FreiHANDLeftHandDataset, batch_size: Optional[int] = 1, shuffle: Optional[bool] = None,
                 sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None, num_workers: int = 0,
                 collate_fn: Optional[_collate_fn_t] = None, pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None, multiprocessing_context=None,
                 generator=None, *, prefetch_factor: int = 2, persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        """
        Initialize an instance of FreiHANDDataLoader.
        It basically specializes input dataset to FreiHANDDataset type.

        :param dataset: dataset from which to load the data.
        :param batch_size: how many samples per batch to load
            (default: ``1``).
        :param shuffle: set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        :param sampler: defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        :param batch_sampler: like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        :param num_workers: how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        :param collate_fn: merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        :param pin_memory: If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        :param drop_last: set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        :param timeout: if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        :param worker_init_fn: If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        :param generator: If not ``None``, this RNG will be used
            by RandomSampler to generate random indexes and multiprocessing to generate
            `base_seed` for workers. (default: ``None``)
        :param prefetch_factor: Number of batches loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers batches prefetched across all workers. (default: ``2``)
        :param persistent_workers: If ``True``, the data loader will not shutdown
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)
        :param pin_memory_device: the data loader will copy Tensors
            into device pinned memory before returning them if pin_memory is set to true.
        """
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)


class AlexNet(nn.Module):
    """
    AlexNet model architecture for hand pose estimation.

    This class implements the AlexNet architecture for hand pose estimation.
    """

    def __init__(self, num_classes=10):
        """
        Initializes AlexNet class

        :param num_classes: number of classes for classification
        """

        super(AlexNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor.
        :return: Output tensor.
        """

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class LeftHandCollectionHandInference(HandCollectionInference):
    """
    A specialized class for managing a collection of both left and right hands
    and performing inference on hand pose estimation results.

    This class extends the HandCollectionInference class and provides methods for loading two trained models,
    performing batch inference on multiple hand images, and retrieving InferenceHand instances for result interpretation.

    Attributes:
    - classification_model: Trained Left-Right hand classification model.
    - estimation_model: Trained HandPoseEstimation network model.
    """

    # CONSTRUCTOR

    def __init__(self, classifier_config: Dict, estimator_config: Dict):
        """
       Initialize a LeftHandCollectionHandInference instance.

       :param classifier_config: model configurations for left vs. right hand classifier.
       :param estimator_config: model configurations for 2-hand pose estimation.
       """

        super().__init__(config=estimator_config)

        ann = AlexNet(num_classes=2)

        self._lr_model = load_model(model=ann, config=classifier_config)

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Get string representation for LeftHandCollectionHandInference object

        :return: string representation for the object
        """

        return f"LeftHandCollectionHandInference"

    # ITEMS

    def __getitem__(self, idx: int) -> tuple[InferenceHand, InferenceHand]:
        """
        Retrieve two InferenceHand instance for result interpretations,
         the first one doesn't apply the classifier, the second one does.

        :param idx: index for selecting a hand from the collection.
        :return: InferenceHand instance for the selected hand.
        """

        # Create collection
        collection = HandCollection()
        hand1 = collection[idx]
        hand2 = collection[idx]

        # Mirror the end to make it left
        hand1 = hand1.mirror()
        hand2 = hand2.mirror()

        # Look if the hand is left
        is_left = hand2.predict_left_hand(model=self._lr_model)

        # Mirror if left
        if is_left:
            hand2 = hand2.mirror()

        pred_heatmaps1 = hand1.predict_heatmap(model=self._model)
        pred_heatmaps2 = hand2.predict_heatmap(model=self._model)

        # Create InferenceHand

        inference_hand1 = InferenceHand(
            idx=idx,
            image=hand1.image,
            keypoints=hand1.keypoints,
            pred_heatmaps=pred_heatmaps1
        )

        inference_hand2 = InferenceHand(
            idx=idx,
            image=hand2.image,
            keypoints=hand2.keypoints,
            pred_heatmaps=pred_heatmaps2
        )

        # Mirror if left
        if is_left:
            inference_hand2 = inference_hand2.mirror()

        return inference_hand1, inference_hand2