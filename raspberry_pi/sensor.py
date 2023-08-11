import Adafruit_DHT

sensor = Adafruit_DHT.DHT11
pin = 4

humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)

if humidity is not None and temperature is not None:
    print(f'Temp={temperature:0.1f}*C  Humid={humidity:0.1f}%')
else:
    print('Failed to get reading. Try again!')