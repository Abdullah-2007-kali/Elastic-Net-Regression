import numpy as np

x = [1, 2, 3]
y = [2, 3, 4]

w = 0.5  # بداية صغيرة لتفادي مشكلة sign(0)
b = 0
alpha = 0.1   # معدل التعلم
lmbda = 0.6   # معامل العقوبة
elastic_alpha = 0.5  # للتحكم بين L1 و L2 (0=L2، 1=L1)

for epoch in range(100):
    total_error_w = 0
    total_error_b = 0
    mse = 0

    for i in range(len(x)):
        y_p = b + w * x[i]
        error = y_p - y[i]
        total_error_w += error * x[i]
        total_error_b += error
        mse += error ** 2

    mse /= (2 * len(x))  # متوسط الخطأ التربيعي
    penalty = lmbda * (elastic_alpha * abs(w) + (1 - elastic_alpha) * (w ** 2))
    total_loss = mse + penalty

    # حساب التدرجات مع Elastic Net
    grad_w = (1 / len(x)) * total_error_w + lmbda * (elastic_alpha * np.sign(w) + 2 * (1 - elastic_alpha) * w)
    grad_b = (1 / len(x)) * total_error_b

    # تحديث المعاملات
    w -= alpha * grad_w
    b -= alpha * grad_b

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: w={w:.4f}, b={b:.4f}, Loss={total_loss:.4f}")

print(f"\n✅ Final result: w={w:.4f}, b={b:.4f}")
