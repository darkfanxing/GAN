from numpy import ndarray, random

def generate_batch_random_images_with_mask(images: ndarray) -> ndarray:
    images_with_mask = images.copy()
    mask = random.randint(
        low=0,
        high=images.shape[1],
        size=int(images_with_mask.shape[1]/4)
    )
    
    images_with_mask[:, mask, :] = 0
    return images_with_mask