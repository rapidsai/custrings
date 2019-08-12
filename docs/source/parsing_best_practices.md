# nvstrings Parsing Best Practices


## Regex Performence Tips:
When working with strings on GPUs, performance is heavily dependent on data and the cuStrings functions you use, especially when doing character/substring substitution and removal.

If you need to write on regex for your pre-processing tasks, follow the below guidelines to achieve best performance.

### Sorting by length:

If your data has unequal length distribution (For example, the longest string might be 1000x longer than the shortest string) and you are doing heavy regex heavy operations, sorting the strings can help to improve the performance.

##### Why:

As regexes operate one character at a time, longer length strings take longer to evaluate and operating on strings in parallel improves throughput.   The GPU works most efficiently when adjacent threads in a warp work in lock step. Wildly different string lengths within adjacent threads means that the threads operating on shorter strings become idle, reducing utilization.

Sorting helps to obtain better wrapping helping adjacent threads in a warp work in lock step to improve utilization . 

### Sorting Alphabetically

Sorting the strings alphabetically can also provide performance gains. 

##### Why:

As each character is processed using a regex, parallel branches may develop causing divergence and reducing efficiency.  Sorting the strings alphabetically can help to reduce this divergence and thus improve the throughput.



These two tenets of parallel computing mean sorting input strings by length can improve utilization, and sorting the strings alphabetically can improve throughput.

### Example:

Lets walk throw a simple example to see the performance differences for ourselves. 

Lets take a simple example where we have to remove `-` character between 2 alphabetical characters while keeping the `-` between numerical characters on a dataset with 4 million short strings and 50k long strings . 


```python
import random
import cudf

small_string = "Exmaple-string 123-456-789"
long_string =  "Exmaple-long-string 123-456-789 "*100
example_list = [small_string]*4_000_000 + [long_string]*50_000
random.shuffle(example_list)

unsorted_sr = cudf.Series(example_list)
%time unsorted_sr.str.replace_with_backrefs('([a-z])-([a-z])',r'\1 \2')
```

```python
CPU times: user 2.58 s, sys: 800 ms, total: 3.38 s
Wall time: 3.38 s
<cudf.Series nrows=4050000 
```

It takes `3.38 s` for processing them.

Lets see how we perform if we just had the `4 million` short strings .

```python
short_string_sr =  cudf.Series([small_string]*4_000_000)
%time short_string_sr.str.replace_with_backrefs('([a-z])-([a-z])',r'\1 \2')
```

```python
CPU times: user 43.1 ms, sys: 19.8 ms, total: 62.9 ms
Wall time: 62.9 ms
<cudf.Series nrows=4000000 >
```

It only takes `62.9` ms if we just had the short strings. 

Lets check our performence on sorted strings: 

```python
text_sr = cudf.Series(example_list)
sorted_ser = text_sr.str.sort(3)
%time sorted_ser.str.replace_with_backrefs('([a-z])-([a-z])',r'\1 \2')
```

```python
CPU times: user 174 ms, sys: 63.5 ms, total: 237 ms
Wall time: 237 ms
<cudf.Series nrows=4050000 >
```

*We see we can get a `14x` boost if we sort our strings by length*
 

## Character and Substring Substitution/Removal


A lot of times during parsing you want to remove substrings and characters . For example, In our post, [show me the word count](https://medium.com/rapids-ai/show-me-the-word-count-3146e1173801), we needed to remove a long list of punctuation characters, as we didn’t want the punctuation characters to impact unique string (word) counts. 



Let’s look at various options of removing punctuation characters on the below example:

```python
import random
import cudf
import re

small_string = "word1!word2,word3<word4>-word5>word7~word8+word9:word10[word11]"
long_string =  small_string*100
example_list = [small_string]*5_000_000 + [long_string]*50_000
random.shuffle(example_list)

cudf_sr = cudf.Series(example_list)

filters = ['!','_',',','{','}','<','>','~','+',':','[',']','-']
```

### Using a for loop:

We could replace them one-by-one in many calls to `replace`, a common first approach to parsing semi-structured logs, but that would be a lot of repetitive function calls and can be slow. 

```python
%%time
for filter_char in list(map(re.escape,filters)):
    cudf_sr = cudf_sr.str.replace(filter_char, ' ')
```

```python
CPU times: user 8.25 s, sys: 2.39 s, total: 10.6 s
Wall time: 10.7 s
```


### Using regex:
We can also use a regular expression (regex) for the replacement. 


#### Regex with capturing groups:

Our first thought may be to use a pattern like  `(!)|(,)...|(%)`    for the character replacement.

```python
filters_regex_capturing_group = '|'.join([f'({ch})' for ch in  map(re.escape, filters)])
print(filters_regex_capturing_group)
%time cudf_sr = cudf_sr.str.replace(filters_regex_capturing_group, ' ')
```

```python
(!)|(_)|(,)|(\{)|(\})|(<)|(>)|(\~)|(\+)|(:)|(\[)|(\])|(\-)
CPU times: user 17.8 s, sys: 5.51 s, total: 23.3 s
Wall time: 23.3 s
```

#### Regex without capturing groups:

Above we used [capturing groups](https://www.regular-expressions.info/brackets.html) to do the regex replacment which we  dont really need when doing a replacment without back references. If we use non capturing groups instead we can get almost a 1.75x  boost in our example: 


```python
filters_regex_without_capturing_group =  '|'.join([f'(?:{ch})' for ch in  map(re.escape, filters)])
print(filters_regex_without_capturing_group)
%time cudf_sr = cudf_sr.str.replace(filters_regex_without_capturing_group, ' ')
```

```python
(?:!)|(?:_)|(?:,)|(?:\{)|(?:\})|(?:<)|(?:>)|(?:\~)|(?:\+)|(?::)|(?:\[)|(?:\])|(?:\-)
CPU times: user 10.1 s, sys: 3.19 s, total: 13.3 s
Wall time: 13.3 s
```



### Using `nvtext`

`nvtext` provides out-of-the-box functions which make these computations like removing multiple characters and stop words much faster.


```python
%time cudf_sr = cudf_sr.str.replace_multi(filters, ' ',regex=False)
```

```python
CPU times: user 724 ms, sys: 230 ms, total: 954 ms
Wall time: 959 ms
```

*Using `nvtext` is thus almost 10x+ faster than the other approaches shown here.*


## Splitting strings for better performance

In `nvstrings`, each operation runs in parallel threads and each string is assigned to a thread. The more strings, the more GPU threads and thus better possible parallelization. 

As a rule of thumb, for the best performance on GPU's, try to keep the average length of strings less and the number of strings large. 

Split functions come that come in handy to do this are:

In case you only want indivsual `tokens` :
* use: `nvtext.tokenize` to convert input text-strings into a long list of nvstrings.

If Number of strings >> Number of tokens per string :
* use: `split` function to split.

If Number of tokens per string >> Number of strings :
* use: `split_record` function to split.


#### Split vs Tokenize

If you have a `DataFrame` loaded with text lines, you might be tempted to just do `df[‘text’].split(‘\s+’)`. That’s fine with CPU- Memory . However, on GPUs its less than ideal by far. GPUs by nature have less total memory than the host. For instance, the Tesla T4 has 16 GB compared to the 512GB or even 1TB of RAM some workstations have these days. 

When you call split, the GPU must allocate one “column” for each token in the string with the highest token count.


```python
import cudf

text_sr = cudf.Series(['hello world', 'hello there my friend', 'hi'])
text_sr.str.split().to_pandas()
```

```
| 0 | 1     | 2     | 3    | 4      |
|---|-------|-------|------|--------|
| 0 | hello | world | None | None   |
| 1 | hello | there | my   | friend |
| 2 | hi    | None  | None | None   |
```

Given some of the lines are quite long (i.e. have many tokens) in many datasets, It is likely to run into an out of memory situation when using `split` as with split we expand the dimensionality of the dataset to the maximum number of tokens present in a sentence across the dataset. To deal with this, `nvtext` has the tokenize method. It generates a single column of all tokens used in the input strings, reducing memory usage significantly.

```python
text_sr = cudf.Series(['hello world', 'hello there my friend', 'hi'])
tokenized_sr = nvtext.tokenize(text_sr.data)
tokenized_sr

print(tokenized_sr)
```

```python
<nvstrings count=7>
['hello', 'world', 'hello', 'there', 'my', 'friend', 'hi
```
