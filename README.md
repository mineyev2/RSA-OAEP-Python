# RSA-OAEP
RSA-OAEP Encryptor/Decryptor

## Installation
You can clone the whole repository into PyCharm, or you can individually download key_generation.py, rsa.py, decryption.py and encryption.py. Just make sure they are contained within the same folder.

## Usage
First, create a public and private key by running key_generation
Then, you can encrypt any message you would like by running encryption.py:
```python
message = input("input message to encrypt: ")
    message = message.encode('utf-8')

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
## Additional Comments
The idea behind RSA-OAEP Encryption is the asymmetric encryption scheme. This means that a hacker cannot decrypt your message even if they know the encryption key. Only a person who has the decryption key can decrypt a message (ideally). So, if you want someone to send you say a password, you would generate a public and private key, but only share the public key to the other person. Then the other person can encrypt their message by using the public key, but nobody can decrypt it except for you, because you the private key to decrypt the coded message.
For those who want the nitty gritty details, check this website: https://tools.ietf.org/html/rfc8017#section-4.1
