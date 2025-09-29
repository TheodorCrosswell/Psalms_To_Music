# Psalms to Music

The goal of this project is to develop an algorithm for matching tunes of existing hymns to verses in the Bible, allowing for quick production of tunes for Bible passages by checking if any existing tunes fit.

## The idea is that you would take an existing tune, for example, Psalm 126, and apply it to a new passage, for example, Psalm 5:

**Psalm 126 tune by Pastor Jonathan Shelley - [YouTube](https://www.youtube.com/watch?v=VCDdQLm9OUw)**

_1 When the Lord turned again the captivity of Zion, we were like them that dream._

_2 Then was our mouth filled with laughter, and our tongue with singing: then said they among the heathen, The Lord hath done great things for them._

_3 The Lord hath done great things for us; whereof we are glad._

+

**Psalm 5**

_1 Give ear to my words, O Lord, consider my meditation._

_2 Hearken unto the voice of my cry, my King, and my God: for unto thee will I pray._

<br>

**Here is how the lyrics map to each other:**

<br>

_1 When the Lord turned again | the captivity of Zion, | we were like them that dream._

_1 Give ear to my words, | O Lord, consider | my meditation._

<br>

_2 Then was our mouth filled with laughter, | and our tongue with singing:_

_2 Hearken unto the voice of my cry, | my King, and my God:_ <s>for unto thee will I pray.</s>

## Current challenges:

- Syllable counting method:
  - Which method is best for counting syllables (CMUDict, pyphen, or syllable)
  - Should I just use all 3 for best accuracy, and to avoid missing out on potential matches?
- Algorithm:
  - How should the algortihm handle texts that almost match the tune, but not quite?
  - Should it optionally drop words that are not significant (e.g. "Selah" or "Amen")?
  - How should punctuation marks or ends of sentences be processed, to avoid splitting a multi-syllable word over a section that is not meant to be a single word in the tune?
  - How to handle adding filler verses from the same text or other text, or adding repeats to fill in sections that don't match perfectly otherwise?
- Known tunes:
  - Converting the known hymns and psalms already put to music, and other good scripture songs into a proper query representation.
- Data formats:
  - How should the query text be processed, as an array of numbers (e.g. "Give ear to my words, O Lord, consider my meditation" -> [1,1,1,1,1,1,1,3,1,4]), or as something else?

I would love to have help on this project, if anyone would like to collaborate or contribute. Contact me directly if you already know me, or email me at theodor.crosswell@gmail.com
