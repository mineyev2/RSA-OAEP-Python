'''
Steps for creating RSA keys:

1) calculate n = pq where p and q are random prime numbers
    - generate p and q using "cryptographically secure pseudo-random number generator"

2) calculate Carmichaels totient function: phi(n) = lcm((p - 1), (q - 1))
    - phi(n) is actually the number of integer which are less than n and are coprime to n.
    - I think there's a theorem which says that phi(n) = lcm((p - 1), (q - 1)) when working with a number like n.

3) Choose integer e s.t. 1 < e < phi(n), and e is coprime to ph(n); gcd(e, phi(n) = 1
    - Typically e = 2^16 + 1 = 65537. It's a big number but not super big or else calculations might take a while
    - smaller applications use e = 3, 5, 35 instead (make sure 35 is coprime to phi(n))

    - d is found using the extended euclidean algoritm

4) compute d s.t. de (congruent) 1 (mod phi(n)) i.e. de = 1 + x*phi(n) for some integer x
    - in other words, d = 1/e mod phi(n); modular inverse



Encrypting/decrypting:

ciphertext c for some message m: c = m^e mod n
    - m is created through "padding" which converts string message to some cryptographically secure number m
    - m < n or else there would be problems (I think)
    - Seems like OAEP is what most people use

=> m = c^d mod n.
    - then can be reconverted to string using the inverse of the padding scheme.


Other stuff:
    - optimal bits for RSA encryption: 1024 is minimum, 2048 is advised. 3072 is for 2030 and past
    - random number generator vs pseudorandom number generator?


Steps to make encryption better
    - use cryptographically secure pseudorandom nunmber generator (python lbirary called secrets)
    - use big primes (don't really now how big though)
    - use p and q such that they're sufficiently far apart (don't know how that works either)
    - use padding scheme (OAEP)

Steps to make things faster/memory efficient:

Done  Extended euclidean method makes calculating d faster
    - use a faster prime number factorization algorithm (examples include: )
    - use lcm(p - 1, q - 1) instead of phi(n)?
        - don't know how or why this works


'''

'''
Steps:

1) set k0 and k1 (typically k0 = k1 = 128
2) set k = how many bits are in n  (for our case it is 2048)
3) compute r, a number that is k0-bits long
4) Run random oracle function G(r), which expands r from being k0 bits to being n - k0 bits in length
    - random oracle is an oracle that responds to every unique query wiht a random respnose chosen uniformly from its output domain
        - oracle is a theoretical black box. Look up for more information
    - random oracles necessary because the proof of why RSA_OAEP works behind the fact that a randomly generated output is created (it is necessary in RSA_OAEP)
        - even though it seems to work with normal cryptographic hash functions, but there is nothing that deterministically proves otherwise, so random oracle is technically safer
        - debate on whether random oracles are actually possible to create
    - actually wikipedia might be wrong. G(r) is a MGF function (mask generating function)
5) X- m00...0 XOR G(r) (i dont understand this step)

Other stuff
    - An octet is a unit which consists of eight bits (a byte)
        - an octet string is a sequence of bytes.

'''

import random

#secrets is CSPRNG
import secrets

import math
import secrets
import hashlib
import os
import typing

Key = typing.Tuple[int, int]

def generate_secure_primes():

    outputs = [0, 0]


    for i in range(0, 2):
        done = False
        while(not done):
            outputs[i] = secrets.randbits(1022)

            #guarantees that first 2 digits of binary value are 11. Also now p and q are 1024-bit
            #should make into variable so its faster
            outputs[i] += 2**1023 + 2**1022


            #to make it impossible to use fermat's factorization method (i have no idea wth it is but whatever
            if(2**1024 - outputs[i] < 2**512):
                print("error: ")
                print(bin(outputs[i]))
                continue

            #testing if value is actually prime here

            # Corner cases
            if (outputs[i] <= 1 or outputs[i] == 4):
                continue
            if (outputs[i] <= 3):
                continue

            # Find r such that n =
            # 2^d * r + 1 for some r >= 1
            d = outputs[i] - 1
            while (d % 2 == 0):
                d //= 2;

            #print(outputs[i])

            # Iterate given nber of 'k' times
            #iterations should be 50
            #guarantees that probability of p and q being prime is extremely high
            for j in range(50):
                if (miillerTest(d, outputs[i]) == False):
                    break
            else:
                done = True

    return outputs[0], outputs[1]


# Utility function to do
# modular exponentiation because it makes things faster cuz we're cool kids
# It returns (x^y) % p
def power(x, y, p):
    # Initialize result
    res = 1;

    # Update x if it is more than or
    # equal to p
    x = x % p;
    while (y > 0):

        # If y is odd, multiply
        # x with result
        if (y & 1):
            res = (res * x) % p;

            # y must be even now
        y = y >> 1;  # y = y/2
        x = (x * x) % p;

    return res;


# This function is called
# for all k trials. It returns
# false if n is composite and
# returns false if n is
# probably prime. d is an odd
# number such that d*2<sup>r</sup> = n-1
# for some r >= 1
def miillerTest(d, n):
    # Pick a random number in [2..n-2]
    # Corner cases make sure that n > 4
    a = 2 + random.randint(1, n - 4);
    # Compute a^d % n
    x = power(a, d, n);

    if (x == 1 or x == n - 1):
        return True;

    # Keep squaring x while one
    # of the following doesn't
    # happen
    # (i) d does not reach n-1
    # (ii) (x^2) % n is not 1
    # (iii) (x^2) % n is not n-1
    while (d != n - 1):
        x = (x * x) % n;
        d *= 2;

        if (x == 1):
            return False;
        if (x == n - 1):
            return True;

            # Return composite
    return False;


#finding d using the extended euclidean algorithm, using the fact taht e and phi(n) are coprime (I think...)
def extended_euclid_d(e, phi_n):

    r_0 = phi_n
    r_1 = e
    r_2 = None

    s_0 = 1
    s_1 = 0
    s_2 = None

    t_0 = 0
    t_1 = 1
    t_2 = None

    q = None

    while(r_1 != 0):
        #quotient
        q = r_0 // r_1

        r_2 = r_0 % r_1
        s_2 = s_0 - q * s_1
        t_2 = t_0 - q * t_1

        #shifting section (shifting all values and rearranging so that they can continue the next iteration

        #if r_1 is smaller, need to swap so that they get calculated correctly
        if(r_1 < r_2):
            temp = r_1
            r_1 = r_2
            r_2 = temp

        r_0 = r_1
        r_1 = r_2

        s_0 = s_1
        s_1 = s_2
        t_0 = t_1
        t_1 = t_2

    #when doing algorithm, it returns the two minimal pairs for x and y i.e. the closest to 0. So that's why some of them are negative. don't know how to force it to be positive atm

    #need to rewrite to s_0 because otherwise confusing from math
    return t_0


# PKCS#1 v2.0 changed phi(n) to lamda(n), which uses lcm instead of just multiplying like a pleb
def lcm(x, y):
    return x * y / math.gcd(x, y)

#OAEP stuff here

def sha1(m: bytes) -> bytes:
    '''SHA-1 hash function'''
    hasher = hashlib.sha1()
    hasher.update(m)
    return hasher.digest()

def os2ip(x: bytes) -> int:
    '''Converts an octet string to a nonnegative integer'''
    return int.from_bytes(x, byteorder='big')

#i2osp converts a nonnegative integer to an octet string of a specified length
def i2osp(x: int, xlen: int) -> bytes:
    '''Converts a nonnegative integer to an octet string of a specified length'''
    return x.to_bytes(xlen, byteorder='big')

def MGF1(seed: bytes, mlen: int) -> bytes:
    '''MGF1 mask generation function with SHA-1'''
    t = b''
    hlen = 128
    for c in range(0, math.ceil(mlen / hlen)):
        _c = i2osp(c, 4)
        t += sha1(seed + _c)
    return t[:mlen]

def xor(data: bytes, mask: bytes) -> bytes:
    '''Byte-by-byte XOR of two byte arrays'''
    masked = b''
    ldata = len(data)
    lmask = len(mask)
    for i in range(max(ldata, lmask)):
        if i < ldata and i < lmask:
            masked += (data[i] ^ mask[i]).to_bytes(1, byteorder='big')
        elif i < ldata:
            masked += data[i].to_bytes(1, byteorder='big')
        else:
            break
    return masked

def get_key_len(key: Key) -> int:
    '''Get the number of octets of the public/private key modulus'''
    _, n = key
    return n.bit_length() // 8

def RSAOAEPEnc(M: bytes, publicKey: Key) -> bytes:
    hLen = 20
    k = get_key_len(publicKey)
    mLen = len(M)
    assert mLen <= k - hLen - 2

    # when put inside function, check step 1 (length checking)
    lHash = sha1(b'')
    hLen = len(lHash)

    # padding scheme
    ps = b'\x00' * (k - mLen - 2 * hLen - 2)

    DB = lHash + ps + b'\x01' + M

    # r = seed
    seed = os.urandom(hLen)

    dbMask = MGF1(seed, k - hLen - 1)
    maskedDB = xor(DB, dbMask)
    seedMask = MGF1(maskedDB, hLen)
    maskedSeed = xor(seed, seedMask)
    EM = b'\x00' + maskedSeed + maskedDB

    m = os2ip(EM)
    #print("m = " + str(m))

    c = RSAEP(publicKey, m)

    C = i2osp(c, k)

    return C

def RSAOAEPDec(privateKey: Key, C: bytes):
    lHash = sha1(b'')

    # default length for when using sha1
    hLen = 20

    k = get_key_len(privateKey)
    assert len(C) == k

    c = os2ip(C)

    d, n = privateKey
    #print(d)

    #If RSADP outputs "ciphertext representative out of range"(meaning that c >= n), output "decryption error" and stop.
    #later
    m = power(c, d, n)
    #print("m = " + str(m))
    EM = i2osp(m, k)

    #EME-OAEP decoding section

    _, maskedSeed, maskedDB = EM[:1], EM[1:1 + hLen], EM[1 + hLen:]

    seedMask = MGF1(maskedDB, hLen)
    seed = xor(maskedSeed, seedMask)
    dbMask = MGF1(seed, k - hLen - 1)
    DB = xor(maskedDB, dbMask)

    _lHash = DB[:hLen]

    # need to do this later but that's a lot of work for now
    #print(_lHash)
    #print(lHash)
    assert lHash == _lHash
    i = hLen
    while i < len(DB):
        if DB[i] == 0:
            i += 1
            continue
        elif DB[i] == 1:
            i += 1
            break
        else:
            raise Exception()
    M = DB[i:]
    return M

def generateKeys() -> typing.Tuple[Key, Key]:
    p, q = generate_secure_primes()
    print(p)
    print(q)

    e = 65537

    phi_n = (p - 1) * (q - 1)

    d = extended_euclid_d(e, phi_n)

    if (d < 0):
        d += phi_n

    n = p * q

    return ((e, n), (d, n))

def RSAEP(publicKey: Key, m: int) -> int:
    e, n = publicKey
    c = power(m, e, n)
    return c