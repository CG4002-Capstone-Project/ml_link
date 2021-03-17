void loop()   {
    delay(33);

    // Delimiter
    Serial.write("#");

    // Mode -- idle, moving or dancing -- 1,2,3
    Serial.write(String(random(1,4)).c_str());
    Serial.write(" ");

    // EMG value
    Serial.write(String(random(0,256) / 255.0 - 0.5).c_str());
    Serial.write(" ");

    // Acc X, Y, Z
    Serial.write(String(random(0,256) / 255.0 - 0.5).c_str());
    Serial.write(" ");
    Serial.write(String(random(0,256) / 255.0 - 0.5).c_str());
    Serial.write(" ");
    Serial.write(String(random(0,256) / 255.0 - 0.5).c_str());
    Serial.write(" ");

    // Gyr X, Y, Z
    Serial.write(String(random(0,256) / 255.0 - 0.5).c_str());
    Serial.write(" ");
    Serial.write(String(random(0,256) / 255.0 - 0.5).c_str());
    Serial.write(" ");
    Serial.write(String(random(0,256) / 255.0 - 0.5).c_str());
    
    Serial.write("\n");
}  

void setup() {
    Serial.begin(115200);               //initial the Serial
}
