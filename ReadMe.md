Car Racing DQL Agent
Bu proje, Car Racing ortamında çalışan bir Deep Q-Learning (DQL) ajanını içerir. Ajan, OpenAI Gym kütüphanesinin CarRacing ortamında eğitilmiş ve test edilmiştir.

Kurulum
Projeyi kopyalayın:
///
%bash
git clone https://github.com/IrohAmca/DQL_Car_Racing.git
cd DQL_Car_Racing
Gerekli bağımlılıkları yükleyin:
///
%bash
pip install -r requirements.txt
Ajanı eğitin veya çalıştırın:
///
%bash
python RL_Car_Racing.py

Kullanım
RL_Car_Racing.py: DQL ajanını eğiten veya çalıştıran ana dosya.

Dosya Yapısı
RL_Car_Racing.py: DQL ajanını eğiten veya çalıştıran ana dosya.
Car_Racing_Agent.py: DQL ajanının sınıfını içeren dosya.
requirements.txt: Bağımlılıkların listesi.

Parametreler
gamma: İndirim faktörü.
learning_rate: Öğrenme oranı.
epsilon_decay: Epsilon değerinin azalma oranı.
epsilon_min: Minimum epsilon değeri.
maxlen: Bellekteki hatırlanan geçmiş gözlemler için maksimum uzunluk.

Katkıda Bulunma
Bu depoyu çatallayın.
Yeni bir dal oluşturun: git checkout -b yeni-dal
Değişikliklerinizi yapın ve bunları işleyin: git commit -m 'Yeni özellik ekle'
Dalınızı ana depo ile birleştirin: git push origin yeni-dal
