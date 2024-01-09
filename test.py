import torchxrayvision as xrv
import skimage, torch, torchvision

img = skimage.io.imread("/tmp/san_data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg")
print(img, img.shape)
img = xrv.datasets.normalize(img, 255, reshape=True)
print(img, img.shape)

img = img.mean(0)[None, ...]
print(img, img.shape)


transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

img = transform(img)
img = torch.from_numpy(img)

# Load model and process image
model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
outputs = model(img[None,...]) # or model.features(img[None,...]) 

print(dict(zip(model.pathologies,outputs[0].detach().numpy())))