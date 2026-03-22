import os


def count_images(path):
    return sum(len(files) for _, _, files in os.walk(path))


train_path = "/Users/puja/IdeaProjects/charamelFaceDetection/data/set1_26_june/train"
test_path = "/Users/puja/IdeaProjects/charamelFaceDetection/data/set1_26_june/test"

train_count = count_images(train_path)
test_count = count_images(test_path)

total = train_count + test_count

print("Train images:", train_count)
print("Test images:", test_count)
print("Train %:", round(train_count / total * 100, 2))
print("Test %:", round(test_count / total * 100, 2))
