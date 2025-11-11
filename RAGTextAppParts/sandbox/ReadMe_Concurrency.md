Read this page 
https://realpython.com/python-concurrency/

# Multiprocessing 
Run code on separate CPU cores -- this gives each process its own memory space and its own python interpreter. save this for when you have large amounts to data to compute

not Global interpreter (GIL) bound

# Threading
Run things on the same core but use multiple threads to do it. so you can do sensor i/o and file i/o things. like one thread is always sending http requests, maybe another thread is always reading from your web cam, and another thread is always reading from the mic.   

threading is prefered when you have to wait around a lot --> but it is GIL bound

# asyncio
use a single thread, but explicitly tell the code when to wait. this is not GIL bound...  

# notes on the text parsing example
Custom Trie Data structure is there just to understand how that data structure works. 

you can speed up word checks with set() --> the "in" keyword is used in loops, but is overloaded to act like .contains() method

In WordCheck.py - i've got a data strucutre (class with methods) -- i turn this into a list of that data structure instead of passing around the dictionary... the data structure when you print it out is just a bunch of memory adresses. 

Also in WordCheck.py  -- im using asyncio -- it's a single thread, but Im telling it not to wait for every itteration of the for loop. Im saying, yeah run the for loop, but dont wait till the previous itteration is done - you have an independant set of instructions so definitely dont wait. 

The other thing I'm doing -- in WordCheck i'm  only reading in one line at a time from the text file -- dont need to hold the whole thing in memory. in the AC_new* examples, I'm readinig in the whole text file so that I can specify window length. 

you could probably use a queue to collect all the information or something if they were dependant, and just  run a second loop. 

