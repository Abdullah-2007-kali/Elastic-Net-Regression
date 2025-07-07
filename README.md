# Elastic-Net-Regression
Elastic Net Regression-model_1

## 📄 **Elastic Net Regression (الانحدار الشبكي المرن)**

### 📌 ما هو Elastic Net Regression؟

هو نموذج انحدار خطي يقوم بدمج ميزات كل من:
✅ **Lasso Regression (L1 Regularization):** الذي يقلل الأوزان ويجعل بعضها = صفر (اختيار المميزات).
✅ **Ridge Regression (L2 Regularization):** الذي يقلل الأوزان الكبيرة بدون تصفيرها بالكامل.

✨ باستخدام **Elastic Net**، نحصل على التوازن المثالي بينهما للتحكم في تعقيد النموذج وتقليل **Overfitting**.

---

### 🧮 **المعادلة الأساسية**

#### 1️⃣ دالة الانحدار:

$$
\hat{y} = b + w_1x_1 + w_2x_2 + ... + w_nx_n
$$

---

#### 2️⃣ دالة الخسارة مع Elastic Net:

$$
\text{Loss} = \underbrace{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}_{\text{MSE}} + \lambda \cdot \left[\alpha \sum |w_j| + (1-\alpha)\sum w_j^2\right]
$$

✅ **MSE**: متوسط الخطأ التربيعي.
✅ $\lambda$: معامل العقوبة (Regularization strength).
✅ $\alpha$: موازنة بين L1 و L2:

* إذا $\alpha=1$ → يصبح النموذج Lasso.
* إذا $\alpha=0$ → يصبح النموذج Ridge.
* إذا $0<\alpha<1$ → نحصل على Elastic Net.

---

### ⚙️ **آلية عمل Elastic Net**

1. يتم حساب الخطأ بين القيم الحقيقية $(y)$ والتنبؤات $(\hat{y})$.
2. يتم إضافة عقوبة على الأوزان:

   * **L1** تجبر بعض الأوزان على الصفر.
   * **L2** تقلل من الأوزان الكبيرة.
3. يتم تحديث المعاملات $w, b$ باستخدام **Gradient Descent**.

---

### 🖋️ **كود Elastic Net Regression من الصفر (Python)**

```python
import numpy as np

# بيانات التدريب
x = [1, 2, 3]
y = [2, 3, 4]

# المعاملات
w = 0.5  # وزن ابتدائي صغير
b = 0    # ثابت ابتدائي
alpha = 0.1        # معدل التعلم
lmbda = 0.6        # قوة العقوبة
elastic_alpha = 0.5  # 0=L2, 1=L1

# التدريب
for epoch in range(100):
    total_error_w = 0
    total_error_b = 0
    mse = 0

    for i in range(len(x)):
        y_pred = b + w * x[i]
        error = y_pred - y[i]
        total_error_w += error * x[i]
        total_error_b += error
        mse += error ** 2

    mse /= (2 * len(x))  # متوسط الخطأ التربيعي
    penalty = lmbda * (elastic_alpha * abs(w) + (1 - elastic_alpha) * (w ** 2))
    total_loss = mse + penalty

    # التدرجات مع Elastic Net
    grad_w = (1 / len(x)) * total_error_w + lmbda * (
        elastic_alpha * np.sign(w) + 2 * (1 - elastic_alpha) * w)
    grad_b = (1 / len(x)) * total_error_b

    # تحديث المعاملات
    w -= alpha * grad_w
    b -= alpha * grad_b

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: w={w:.4f}, b={b:.4f}, Loss={total_loss:.4f}")

print(f"\n✅ النتيجة النهائية: w={w:.4f}, b={b:.4f}")
```

---

### 📊 **مميزات Elastic Net**

✅ يجمع بين ميزات Lasso و Ridge.
✅ يتعامل مع البيانات عالية الأبعاد (High-dimensional data).
✅ يقلل من Overfitting ويحسن التعميم (Generalization).
✅ مناسب عندما تكون هناك مميزات متعددة مترابطة بشدة.

---

### 📌 **الخلاصة**

* Elastic Net هو مزيج ذكي بين **Lasso** و **Ridge**.
* باستخدام معامل $\alpha$، يمكنك تخصيص مقدار الاعتماد على L1 أو L2 حسب مشكلتك.
* عند $\alpha=1$ يصبح Lasso، وعند $\alpha=0$ يصبح Ridge.

---

## 🌟 **إن أعجبك المشروع لا تنسَ دعمنا بـ Star ⭐️ على GitHub!**
