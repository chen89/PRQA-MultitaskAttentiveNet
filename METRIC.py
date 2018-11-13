# def mertic(prediction, y, threshold):
#     tp = 0
#     tn = 0
#     fp = 0
#     fn = 0
#     prediction = [1 if pred >= threshold else 0 for pred in prediction]
#     for pr, tr in zip(prediction, y):
#         if pr == 1 and tr == 1:
#             tp += 1
#         if pr == 1 and tr == 0:
#             fp += 1
#         if pr == 0 and tr == 1:
#             fn += 1
#         if pr == 0 and tr == 0:
#             tn += 1
#
#     precision = round(float(tp) / (tp + fp), 6)
#     recall = round(float(tp) / (tp + fn), 6)
#     f1 = round(2 * (precision * recall) / (precision + recall), 6)
#     return precision, recall, f1

def mertic(prediction, y, threshold):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    prediction = [1 if pred >= threshold else 0 for pred in prediction]
    for pr, tr in zip(prediction, y):
        if pr == 1 and tr == 1:
            tp += 1
        if pr == 1 and tr == 0:
            fp += 1
        if pr == 0 and tr == 1:
            fn += 1
        if pr == 0 and tr == 0:
            tn += 1

    if (tp + fn) != 0 and (tp + fp) != 0 and tp != 0:
        precision = round(float(tp) / (tp + fp), 6)
        recall = round(float(tp) / (tp + fn), 6)
        f1 = round(2 * (precision * recall) / (precision + recall), 6)
    else:
        precision = 0
        recall = 0
        f1 = 0
    return precision, recall, f1

