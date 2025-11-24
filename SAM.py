class SAM():
    """Class to initialize and use the Segment Anything Model (SAM) for mask generation."""
    
    def __init__(self, path="sam_vit_b_01ec64.pth"):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2

        self.sam = sam_model_registry["vit_b"](checkpoint=path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
    
    def get_masks(self, image):
        """Generate masks from input image using SAM's mask generator."""
        # Convert image if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        masks = self.mask_generator.generate(image)
        return masks   

    

    def get_anns(self, anns):
        """Convert SAM masks to visualization format."""
        import numpy as np
        import matplotlib.pyplot as plt
        
        if len(anns) == 0:
            return None
            
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        return img

    def plot_masks(self, image, anns):
        """Plot the image with the generated masks overlaid."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        ax[0].imshow(image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(image)
        ax[1].imshow(anns)
        ax[1].set_title('SAM Masks')
        ax[1].axis('off')
        plt.show()

# Fixed example usage
import cv2
import matplotlib.pyplot as plt

# Initialize SAM
sam = SAM()

# Load and convert image properly
image = cv2.imread('mvtec_anomaly_detection/capsule/test/crack/001.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Get masks and annotations
masks = sam.get_masks(image)
anns = sam.get_anns(masks)    

# Plot results
sam.plot_masks(image, anns)