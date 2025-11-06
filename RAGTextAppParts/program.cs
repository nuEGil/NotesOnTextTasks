using System;
using System.IO;
using System.Runtime.InteropServices.Swift;

class Test
{// static methods are parts of the class itself -- non-static only belong to a specific instance. 
    public static void DataDump(string output_path)
    {
        // Write a timestamp into this thing too
        string timestamp = DateTime.Now.ToString();
        if (!File.Exists(output_path))
        {
            //Create a file to write to . 
            using (StreamWriter sw = File.CreateText(output_path))
            {
                // Trying to write the time stamp
                sw.WriteLine($"Timestamp: {timestamp}");
                sw.WriteLine("Information: There is something to write down");
                sw.WriteLine("Number: 256");
            }
        }
    } 
    public struct LogEntry
    {
        public LogEntry(string text, string time, string[] keywords)
        {
            Text = text;
            Created = time;
            Keywords = keywords;
        }
        public string Text {get; init;}
        public string Created {get; init;}
        public string[] Keywords {get; init;}
        
        // if you want to print this thing to the console 
        public override string ToString()
        {
            string output = $"Text:{Text}\nCreated:{Created}\nKeywords:[";
            foreach(string word in Keywords)
            {
                output+=$"{word},";
            }
            output+="]\n";
            return output; 
        }
    }
    public static string[] keyWordCheck(string text)
    {
        // sample string array of things to look out for
        string[] keywords = [
                       "love", "passion", "faith", "war",
                       "time","seconds", "minutes", "days", "weeks",
                       "months", "years", "hours", 
                       "mind", "mental", "memory", "memories",
                       "think", "thinking", "thought", "thoughts",
                       "prayers", "pray", "prayed", "meditate",
                       "soul", "spirit", "dream", 
                       "body", "bodies", "blood", "guts", "spit", 
                       "vomit", "excrement", "soiled",
                       "eyes", "mouth", "nose", "ears", "hands",
                       "hand", "fingers", "finger", "feet", "toes",
                       "heart", "stomach", "nerve", "nervous",
                       "neck", "chest", "breast", "shoulders",
                       "kill", "killed","murder", "suicide",
                       "death", "died", "dead",
                       "drunk", "drink", 
                       "drinking", "smoking", 
                       "gambling", "gamble","gambler", "alcoholic",
                       "habit", "drug", "drugs","opium", "cocaine",
                       "family","father", "mother", "brother", "sister",
                       "aunt", "uncle", "cousin", "son", "daughter",
        ];
        string tags = "";
        string[] output = ["NA"];
        foreach (string kw in keywords)
        {   
            // text.Contains - is a pattern not exact word match
            if (text.Contains(kw))
            {
                tags += $"{kw} ";
            }

        }
        tags = tags.Trim();

        if (tags.Length > 1)
        {
            output = tags.Split(" ");
        }
        
        return output;
    } 

    public static void LoadBook(string book_path) 
    {
        //Read text out of a file. 
        using (StreamReader sr = File.OpenText(book_path))
        {
            int count = 0; // line counter
            string s; // line placeholder
            string subtext = ""; // sub text
            while ((s = sr.ReadLine()) != null)
            {
                //Thread.Sleep(5); // little delay to get the time stamps to be different
                subtext+= s; // add to the sub text
                count++;
                if (count == 5)
                {   
                    string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
                    // check for key words
                    string[] keywords = keyWordCheck(subtext);
                    var aa = new LogEntry(subtext, timestamp, keywords);
                    Console.WriteLine(aa); // output!
                    count = 0; // reset the count so that we can update this 
                    subtext = ""; // reset the text so that we dont keep adding text
                }
            }
            Console.Write($"Number of book lines {count}");
        }
       
    }

    public static void Main()
    {
        // set up a couple of files to work with
        string note_path = "Your text file here";
        string book_path = "Your text file here";

        // Writing some data to a text file  
        DataDump(note_path);
        // Ok so I can load a book. 
        LoadBook(book_path);
        
    }

}