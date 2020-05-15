# RSA-OAEP
RSA-OAEP Encryptor/Decryptor

##Installation
You can clone the whole repository into PyCharm, or you can individually download key_generation.py, rsa.py, decryption.py and encryption.py. Just make sure they are contained within the same folder.

##Usage
First, create a public and private key by running key_generation
Then, you can encrypt any message you would like by running encryption.py:
```python
message = input("input message to encrypt: ")
    message = message.encode('utf-8')

    # enter public key tuple that was generated from key_generation
    pub_key = #enter public key tuple that was generated from key_generation.py
    C = rsa.RSAOAEPEnc(message, pub_key)
    
    print(C)
```
To decrypt the code, copy the output C and paste it along with the decryption key into decrytion.py:
```python
    C = #enter C generated from running encryption.py
    
    prv_key = #enter private key generated from key_generation
    M = rsa.RSAOAEPDec(prv_key, C)

    print(M)
```
If everything works, then M should equal the initial message.
